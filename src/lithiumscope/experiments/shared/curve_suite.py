"""Persist reproducible whole-group learning-curve diagnostics, never models."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd

from lithiumscope.core.reproducibility import canonical_json_hash, runtime_fingerprint
from lithiumscope.core.scientific.checkpoint_identity import evaluation_identity
from lithiumscope.core.scientific.learning_curves import build_curve_plan, curve_plan_records, evaluate_curve


def summarize_curves(table: pd.DataFrame) -> pd.DataFrame:
    """Average seeds inside a fold before aggregating independent outer folds."""
    rows = []
    keys = ["algorithm", "fraction_of_eligible_groups"]
    metrics = [c for c in table if c.startswith(("train_", "test_"))
               and c not in {"train_groups", "test_groups"}]
    for (algorithm, fraction), group in table.groupby(keys, sort=False):
        valid = group[group.status == "ok"]
        row = {"algorithm": algorithm, "fraction_of_eligible_groups": fraction,
               "planned_partitions": len(group), "successful_partitions": len(valid),
               "failed_partitions": int(group.status.eq("failed").sum()),
               "unavailable_partitions": int((~group.status.isin(["ok", "failed"])).sum()),
               "planned_folds": group.fold.nunique(), "successful_folds": valid.fold.nunique(),
               "state": "complete" if len(valid) == len(group) else "incomplete"}
        for column in metrics:
            by_fold = valid.groupby("fold")[column].mean().dropna()
            row[column + "_mean"] = float(by_fold.mean()) if len(by_fold) else np.nan
            row[column + "_std_across_folds"] = float(by_fold.std(ddof=1)) if len(by_fold) > 1 else np.nan
            row[column + "_valid_folds"] = len(by_fold)
        rows.append(row)
    return pd.DataFrame(rows)


def run_curve_suite(*, x, y, groups, splits, kind, algorithms, device, policy,
                    output: Path, fit, predict, eligible=None, endpoints=None,
                    execute=False, fractions=(.25, .5, .75, 1.), seeds=(42, 43, 44)):
    """Default is preflight. Fitting requires an explicit diagnostic execution flag.

    fit(algorithm, fold, X_train, y_train, groups_train) must create a fresh
    estimator with a predeclared policy, never receive test observations.
    """
    algorithms = tuple(algorithms)
    splits = list(splits)
    if not algorithms or len(set(algorithms)) != len(algorithms):
        raise ValueError("Curve suite requires distinct algorithm families")
    if not x.index.equals(y.index) or not x.index.equals(groups.index):
        raise ValueError("Curve cohort row identity differs")
    plan = build_curve_plan(y, groups, splits, kind=kind, eligible=eligible,
                            endpoints=endpoints, fractions=fractions, seeds=seeds)
    group = "model_1" if kind == "regression" else "model_2"
    declared = {"kind": kind, "algorithms": list(algorithms), "policy": policy,
                "device": asdict(device), "fractions": list(fractions), "seeds": list(seeds),
                "scope": "diagnostic_only_no_model_publication"}
    identity = evaluation_identity(output, x=x, y=y, groups=groups, splits=splits,
                                   policy=declared, model_group=group,
                                   extra={"eligible": None if eligible is None else np.asarray(eligible).tolist(),
                                          "endpoints": None if endpoints is None else list(endpoints)})
    records = curve_plan_records(plan)
    manifest = {"schema_version": 1, **declared, "identity": identity,
                "plan_sha256": canonical_json_hash(records),
                "runtime": runtime_fingerprint(), "partitions": records,
                "state": "preflight", "fitted_models_saved": False}
    output.mkdir(parents=True, exist_ok=False)
    path = output / "curve_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, default=str, allow_nan=False), encoding="utf-8")
    if not execute:
        return manifest
    tables = []
    start = time.monotonic()
    try:
        for algorithm in algorithms:
            table = evaluate_curve(x, y, groups, plan, kind=kind, algorithm=algorithm,
                                   fit=lambda fold, a, b, g: fit(algorithm, fold, a, b, g),
                                   predict=predict)
            tables.append(table)
            # Save partial evidence after each family, including failures.
            combined = pd.concat(tables, ignore_index=True)
            combined.to_csv(output / "curve_partitions.csv", index=False)
            summarize_curves(combined).to_csv(output / "curve_summary.csv", index=False)
        manifest["state"] = "complete" if all(t.status.eq("ok").all() for t in tables) else "incomplete"
    except BaseException:
        manifest["state"] = "interrupted"
        raise
    finally:
        manifest["completed_families"] = len(tables)
        manifest["elapsed_seconds"] = time.monotonic() - start
        path.write_text(json.dumps(manifest, indent=2, default=str, allow_nan=False), encoding="utf-8")
    return manifest
