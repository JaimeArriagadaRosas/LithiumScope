"""Local opt-in sensitivity study; no changes to active M2 models."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from lithiumscope.core.config import load_config
from lithiumscope.core.reproducibility.hashing import file_sha256
from lithiumscope.core.paths import RESULTS_DIR
from lithiumscope.datasets.management.sources.local import local_training_source
from lithiumscope.experiments.shared.design import choose_algorithms
from lithiumscope.core.scientific.classification_metrics import classification_metrics
from lithiumscope.model_2.training.calibration.operating import fit_policy
from lithiumscope.model_2.training.calibration.policy import calibration_policy
from lithiumscope.model_2.training.execution.competition_prepare_scientific_data import _prepare_scientific_data
from lithiumscope.model_2.validation.sensitivity import POLICIES, plan_protocol
from lithiumscope.resources.detection.device import detect_device


def run(*, output: Path, endpoint_ppm: float, algorithms=None, execute=False):
    config = load_config("model_2")
    model, training, validation = config["model"], config["training"], config["validation"]
    selected = choose_algorithms(model["algorithms"], algorithms)
    seed, folds, inner = int(model["random_seed"]), int(validation["folds"]), int(validation["inner_folds"])
    mode = calibration_policy(config["optimization"])
    source = local_training_source("model_2")
    output.mkdir(parents=True, exist_ok=False)
    preparation = output / "preparation"
    preparation.mkdir()
    _, frame, _, _, _, mask, _, _, x, _, _ = _prepare_scientific_data(
        config, SimpleNamespace(manifests=preparation), model.get("fixed_threshold_ppm"),
        training["lithium_column"], source, float(model["high_lithium_quantile"]),
        folds, seed, training["spatial_group_column"])
    manifest = pd.read_csv(source)
    lithium = frame[training["lithium_column"]].to_numpy(float)
    if len(manifest) != len(frame) or not np.array_equal(manifest.Li_icpms.to_numpy(float), lithium):
        raise ValueError("Spatial study source rows differ from feature matrix")
    if not np.isfinite(endpoint_ppm) or endpoint_ppm < 0:
        raise ValueError("Independent endpoint must be finite and nonnegative")
    if manifest.sample_id.isna().any() or not manifest.sample_id.is_unique:
        raise ValueError("Spatial study requires unique sample identities")
    labels = (lithium >= endpoint_ppm).astype(int)
    plans, groups_by_protocol = [], {}
    for name, grid, buffer_m, scene in POLICIES:
        policy = {"protocol": name, "grid_degrees": grid, "buffer_m": buffer_m, "scene": scene}
        try:
            groups, records = plan_protocol(
                manifest, lithium, grid=grid, buffer_m=buffer_m, combine_scene=scene,
                folds=folds, endpoint_ppm=endpoint_ppm, quality_mask=mask)
        except ValueError as exc:
            plans.append({**policy, "state": "unavailable", "reason": str(exc), "folds": []})
            continue
        groups_by_protocol[name] = groups
        for item in records:
            if item["state"] != "ready":
                continue
            train = item["train"]
            g = groups.iloc[train]
            if g.nunique() < inner:
                item["state"] = "unavailable_inner"
                continue
            for a, b in GroupKFold(n_splits=inner).split(train, groups=g):
                if np.unique(labels[np.asarray(train)[a]]).size != 2 or np.unique(labels[np.asarray(train)[b]]).size != 2:
                    item["state"] = "unavailable_inner"
                    break
        plans.append({**policy, "state": "ready" if all(r["state"] == "ready" for r in records)
                      else "partially_unavailable", "folds": records})
    report = {"state": "preflight", "endpoint_ppm": endpoint_ppm, "source_sha256": file_sha256(source),
              "algorithms": selected, "hyperparameters": "defaults", "seed": seed, "calibration": mode,
              "buffer_scope": "outer_fit_only", "automatic_selection": False, "protocols": plans}
    target = output / "manifest.json"

    def save():
        target.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")

    save()
    if not execute:
        return report
    records, predictions = [], []
    device = detect_device(prefer_gpu=False)
    try:
        for plan in plans:
            name = plan["protocol"]
            groups = groups_by_protocol.get(name)
            for item in plan["folds"]:
                train, test = item["train"], item["test"]
                for algorithm in ["baseline_prior", *selected]:
                    row = {"protocol": name, "fold": item["fold"], "algorithm": algorithm,
                           "state": item["state"], "n_fit": len(train), "n_test": len(test)}
                    if item["state"] == "ready" or (algorithm == "baseline_prior" and item["state"] == "unavailable_inner"):
                        try:
                            if algorithm == "baseline_prior":
                                scores = np.full(len(test), labels[train].mean())
                            else:
                                fitted = fit_policy(algorithm, device, seed + item["fold"], {},
                                                    x.iloc[train], pd.Series(labels[train], index=x.index[train]),
                                                    groups.iloc[train], inner, mode)
                                scores = fitted.predict_proba(x.iloc[test])[:, 1]
                            row.update(classification_metrics(labels[test], scores), state="ok")
                            predictions.extend({"protocol": name, "algorithm": algorithm, "fold": item["fold"],
                                                "sample_id": str(manifest.sample_id.iloc[i]),
                                                "truth": int(labels[i]), "score": float(p)}
                                               for i, p in zip(test, scores, strict=True))
                        except Exception as exc:
                            row.update(state="failed", error_type=type(exc).__name__)
                    records.append(row)
                    pd.DataFrame(records).to_csv(output / "fold_metrics.csv", index=False)
                    pd.DataFrame(predictions).to_csv(output / "oof.csv", index=False)
        summary = []
        table = pd.DataFrame(predictions)
        if not table.empty:
            for (name, algorithm), values in table.groupby(["protocol", "algorithm"]):
                if len(values) == len(manifest) and values.sample_id.is_unique:
                    summary.append({"protocol": name, "algorithm": algorithm, "state": "complete_oof",
                                    **classification_metrics(values.truth, values.score)})
        pd.DataFrame(summary).to_csv(output / "global_metrics.csv", index=False)
        report["state"] = ("complete" if all(p["state"] == "ready" for p in plans)
                           and all(r["state"] == "ok" for r in records) else "incomplete")
    except BaseException:
        report["state"] = "interrupted"
        raise
    finally:
        save()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint-ppm", type=float, required=True,
                        help="Independent predeclared geochemical endpoint; not a TEST-derived quantile")
    parser.add_argument("--algorithm", action="append")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    output = RESULTS_DIR / "model_2" / "methodology" / (
        "spatial_sensitivity_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8])
    report = run(output=output, endpoint_ppm=args.endpoint_ppm, algorithms=args.algorithm, execute=args.execute)
    print(f"{report['state']} — {output}")


if __name__ == "__main__":
    main()
