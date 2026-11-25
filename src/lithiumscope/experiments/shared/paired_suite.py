"""Isolated all-fold experiment runner. Preflight by default, no saved estimators."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from time import monotonic
from uuid import uuid4

import numpy as np
import pandas as pd

from lithiumscope.core.scientific_checks import assert_group_isolation, assert_partition_coverage
from lithiumscope.core.reproducibility import canonical_json_hash, runtime_fingerprint
from lithiumscope.core.scientific.checkpoint_identity import evaluation_identity
from lithiumscope.experiments.shared.paired_comparison import paired_group_intervals


@dataclass(frozen=True)
class Arm:
    name: str
    algorithm: str
    reference: str | None
    x: pd.DataFrame
    policy: dict
    eligible: np.ndarray | None = None


def _write(path, record):
    temp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temp.write_text(json.dumps(record, indent=2, default=str, allow_nan=False), encoding="utf-8")
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def validate_arms(arms, y, groups, splits, kind, endpoints):
    if kind not in {"regression", "classification"}:
        raise ValueError("Unknown experiment kind")
    if not isinstance(y, pd.Series) or not isinstance(groups, pd.Series) or not y.index.equals(groups.index):
        raise ValueError("Experiment target/groups must have matching indices")
    if not y.index.is_unique or not np.isfinite(y.to_numpy(float)).all() or (y < 0).any():
        raise ValueError("Invalid experiment target")
    if groups.isna().any() or groups.astype(str).str.strip().eq("").any():
        raise ValueError("Invalid spatial groups")
    assert_partition_coverage(splits, len(y))
    for train, test in splits:
        assert_group_isolation(train, test, groups)
    if kind == "classification":
        if endpoints is None or len(endpoints) != len(splits) or not np.isfinite(endpoints).all() or min(endpoints) < 0:
            raise ValueError("Frozen TRAIN endpoints required")
    elif endpoints is not None:
        raise ValueError("Unexpected classification endpoints")
    names = {arm.name: arm for arm in arms}
    if not arms or len(names) != len(arms):
        raise ValueError("Experiment arm names must be unique")
    masks = {}
    for arm in arms:
        if (not arm.x.index.equals(y.index) or not arm.x.columns.is_unique or not len(arm.x.columns)
                or any(str(c).strip().lower() in {"li_icpms", "lithium", "target", "spatial_group"} for c in arm.x)):
            raise ValueError("Experiment matrix identity or predictor contract differs")
        if arm.reference is not None:
            if arm.reference not in names or names[arm.reference].reference is not None or names[arm.reference].algorithm != arm.algorithm:
                raise ValueError("Invalid paired reference")
        mask = np.ones(len(y), dtype=bool) if arm.eligible is None else np.asarray(arm.eligible)
        if mask.shape != (len(y),) or mask.dtype.kind != "b" or not mask.any():
            raise ValueError("Invalid TRAIN eligibility")
        masks[arm.name] = mask
    return masks


def _metrics(kind, truth, prediction):
    if kind == "regression":
        from lithiumscope.core.scientific.regression_metrics import regression_metrics
        return regression_metrics(truth, prediction).to_dict()
    from lithiumscope.core.scientific.classification_metrics import classification_metrics
    return classification_metrics(truth, prediction)


def run_paired_suite(*, arms, y, groups, splits, kind, output: Path, fit, predict,
                     endpoints=None, execute=False, bootstrap_repeats=500):
    arms, splits = list(arms), list(splits)
    masks = validate_arms(arms, y, groups, splits, kind, endpoints)
    group = "model_1" if kind == "regression" else "model_2"
    plan, identities = [], {}
    for arm in arms:
        identities[arm.name] = evaluation_identity(output, x=arm.x, y=y, groups=groups, splits=splits,
            policy={"arm": arm.name, "algorithm": arm.algorithm, "policy": arm.policy},
            extra={"eligible": masks[arm.name].tolist(), "endpoints": endpoints}, model_group=group)
        for fold, (train, test) in enumerate(splits, 1):
            train, test = np.asarray(train), np.asarray(test)
            train = train[masks[arm.name][train]]
            cut = None if endpoints is None else float(endpoints[fold-1])
            labels = y if cut is None else (y >= cut).astype(int)
            ready = len(train) >= 2 and groups.iloc[train].nunique() >= 2
            if cut is not None:
                ready = ready and labels.iloc[train].nunique() == labels.iloc[test].nunique() == 2
            plan.append({"arm": arm.name, "fold": fold, "train": train.tolist(), "test": test.tolist(),
                         "endpoint_ppm": cut, "state": "ready" if ready else "unavailable"})
    report = {"schema_version": 1, "state": "preflight", "kind": kind,
              "arms": [{"name": a.name, "algorithm": a.algorithm, "reference": a.reference,
                        "policy": a.policy, "identity": identities[a.name]} for a in arms],
              "plan_sha256": canonical_json_hash(plan), "partitions": plan,
              "runtime": runtime_fingerprint(), "fitted_models_saved": False,
              "automatic_selection": False, "external_validation": False}
    output.mkdir(parents=True, exist_ok=False)
    manifest = output / "manifest.json"
    _write(manifest, report)
    if not execute:
        return report
    records, predictions = [], []
    by_name = {a.name: a for a in arms}
    start = monotonic()
    try:
        for item in plan:
            arm = by_name[item["arm"]]
            train, test = item["train"], item["test"]
            labels = y if item["endpoint_ppm"] is None else (y >= item["endpoint_ppm"]).astype(int)
            row = {"arm": arm.name, "algorithm": arm.algorithm, "fold": item["fold"],
                   "n_train": len(train), "n_test": len(test), "endpoint_ppm": item["endpoint_ppm"],
                   "state": item["state"]}
            began = monotonic()
            if row["state"] == "ready":
                try:
                    model = fit(arm, item["fold"], arm.x.iloc[train], labels.iloc[train],
                                groups.iloc[train], y.iloc[train])
                    train_prediction = np.asarray(predict(model, arm.x.iloc[train]), dtype=float)
                    test_prediction = np.asarray(predict(model, arm.x.iloc[test]), dtype=float)
                    row.update({f"train_{k}": v for k,v in _metrics(kind, labels.iloc[train], train_prediction).items()})
                    row.update({f"test_{k}": v for k,v in _metrics(kind, labels.iloc[test], test_prediction).items()})
                    predictions.extend({"arm": arm.name, "position": int(i), "fold": item["fold"],
                                        "group": str(groups.iloc[i]), "truth": float(labels.iloc[i]),
                                        "prediction": float(p)} for i,p in zip(test,test_prediction,strict=True))
                    row["state"] = "ok"
                except Exception as exc:
                    row.update(state="failed", error_type=type(exc).__name__)
            row["elapsed_seconds"] = monotonic()-began
            records.append(row)
            pd.DataFrame(records).to_csv(output / "fold_metrics.csv", index=False)
            pd.DataFrame(predictions, columns=["arm","position","fold","group","truth","prediction"]).to_csv(
                output / "paired_predictions.csv", index=False)
        table = pd.DataFrame(predictions, columns=["arm","position","fold","group","truth","prediction"])
        intervals = []
        for arm in arms:
            if arm.reference is None:
                continue
            a, b = table[table.arm == arm.reference], table[table.arm == arm.name]
            if len(a) != len(y) or len(b) != len(y):
                intervals.append({"arm": arm.name, "reference": arm.reference, "state": "incomplete_oof"})
                continue
            for result in paired_group_intervals(a,b,kind=kind,repeats=bootstrap_repeats):
                intervals.append({"arm": arm.name,"reference":arm.reference,"state":"ok",**result})
        pd.DataFrame(intervals).to_csv(output / "paired_intervals.csv", index=False)
        report["state"] = "complete" if all(r["state"] == "ok" for r in records) else "incomplete"
    except BaseException:
        report["state"] = "interrupted"
        raise
    finally:
        report["elapsed_seconds"] = monotonic()-start
        report["completed_partitions"] = len(records)
        _write(manifest, report)
    return report
