"""Nested, paired development screening. No release artifacts are written."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from uuid import uuid4

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score

from lithiumscope.core.config import load_config
from lithiumscope.core.paths import RESULTS_DIR
from lithiumscope.experiments.generalization.arms import M1_ARMS, M2_ARMS, classification_fit, regression_fit
from lithiumscope.experiments.generalization.prepared import load_snapshot, prepare_version
from lithiumscope.experiments.generalization.partitions import partitions
from lithiumscope.experiments.generalization.preflight import require_treatments
from lithiumscope.resources.detection.device import detect_device


def metrics(group, y, scores):
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite predictions")
    if group == "model_1":
        return {"rmse": float(np.sqrt(mean_squared_error(y, scores))), "mae": float(mean_absolute_error(y, scores)),
                "r2": float(r2_score(y, scores)), "negative_predictions": int((scores < 0).sum())}
    return {"roc_auc": float(roc_auc_score(y, scores)), "ap": float(average_precision_score(y, scores))}


def evaluate(group, algorithm, arm, device, seed, data, train, test, cfg):
    x, y, groups, sources, eligible, numeric, categorical = data
    train, test = np.asarray(train), np.asarray(test)
    if np.intersect1d(train, test).size or set(groups.iloc[train]) & set(groups.iloc[test]):
        raise ValueError("Spatial leakage in experiment")
    real_train = train.copy()
    train = train[eligible[train]]
    if not len(train):
        raise ValueError("No QA eligible TRAIN")
    if group == "model_1":
        fit_y, test_y = y.iloc[train], y.iloc[test]
        pred, train_pred, provenance = regression_fit(algorithm, arm, device, seed, x.iloc[train], fit_y,
            groups.iloc[train], sources.iloc[train], numeric, categorical, x.iloc[test])
    else:
        model = cfg["model"]
        fixed = model.get("fixed_threshold_ppm")
        threshold = float(fixed if fixed is not None else y.iloc[real_train].quantile(model["high_lithium_quantile"]))
        fit_y, test_y = (y.iloc[train] >= threshold).astype(int), (y.iloc[test] >= threshold).astype(int)
        if fit_y.nunique() != 2 or test_y.nunique() != 2:
            raise ValueError("Single-class spatial partition")
        pred, train_pred, provenance = classification_fit(algorithm, arm, device, seed, x.iloc[train], fit_y,
                                                        groups.iloc[train], x.iloc[test])
        provenance["threshold_ppm"] = threshold
        from lithiumscope.experiments.generalization.signature_policy import signature_weights
        weights = signature_weights(x.iloc[test])
        provenance["signature_weighted_auc"] = float(roc_auc_score(test_y,pred,sample_weight=weights))
        provenance["signature_weighted_ap"] = float(average_precision_score(test_y,pred,sample_weight=weights))
    from lithiumscope.experiments.generalization.domain import support_summary
    provenance.update(support_summary(x.iloc[train],x.iloc[test],numeric,categorical))
    test_metrics = metrics(group, test_y, pred)
    train_metrics = metrics(group, fit_y, train_pred)
    return {**test_metrics, **{"train_"+k: v for k, v in train_metrics.items()},
            "real_train": len(train), "test_rows": len(test), **provenance}, pred, test_y.to_numpy()


def run_screen(group, source, *, all_families=False, output_root=None, device=None):
    cfg = load_config(group)
    directory, _ = prepare_version(group, source, run_eda=False)
    data = load_snapshot(directory, directory.name)
    x, y, groups, _, _, _, _ = data
    device = device or detect_device(prefer_gpu=False)
    plan_directory, plan_manifest = require_treatments(directory,cfg)
    readiness = pd.read_csv(plan_directory/"treatments.csv")
    outer = partitions(group, data, cfg)
    outer_count, inner_count = len(outer), len(outer[0][2])
    arms = M1_ARMS if group == "model_1" else M2_ARMS
    expected = {(fold,inner,arm) for fold in range(1,outer_count+1)
                for inner in range(inner_count+1) for arm in arms}
    observed = set(zip(readiness.outer_fold,readiness.inner_fold,readiness.arm,strict=True))
    if expected != observed or len(readiness) != len(expected):
        raise ValueError("Treatment readiness differs from current partition contract")
    ready = {(int(row.outer_fold),int(row.inner_fold),row.arm) for row in readiness.itertuples() if row.state == "ready"}
    algorithms = cfg["model"]["algorithms"] if all_families else ["random_forest"]
    from pathlib import Path
    root = Path(output_root) if output_root is not None else RESULTS_DIR / "generalization"
    destination = root / group / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid4().hex[:8])
    destination.mkdir(parents=True)
    results, inner_results, predictions, provenance = [], [], [], []
    metric = "rmse" if group == "model_1" else "roc_auc"
    for algorithm in algorithms:
        for fold, (tr, te, inner) in enumerate(outer, 1):
            candidates = []
            for arm in arms:
                if not all((fold,i,arm) in ready for i in range(inner_count+1)):
                    inner_results.append({"algorithm":algorithm,"outer_fold":fold,"arm":arm,
                                          "state":"blocked_preflight","reason":"Treatment not ready in every TRAIN partition"})
                    continue
                scores = []
                secondary = []
                admissible = True
                for i, (a, b) in enumerate(inner, 1):
                    try:
                        result, _, _ = evaluate(group, algorithm, arm, device, 42+fold*10+i, data, a, b, cfg)
                        scores.append(result[metric])
                        secondary.append(result["mae"] if group == "model_1" else result["ap"])
                        if group == "model_1" and result["negative_predictions"]:
                            admissible = False
                        provenance.extend({"algorithm": algorithm, "fold": fold, "inner_fold": i, "arm": arm, "stage": "inner", **p} for p in result["generated"])
                        inner_results.append({"algorithm": algorithm, "outer_fold": fold, "inner_fold": i,
                            "arm": arm, "state": "ok", **{k:v for k,v in result.items() if k != "generated"}})
                    except Exception as exc:
                        inner_results.append({"algorithm": algorithm, "outer_fold": fold, "inner_fold": i,
                                              "arm": arm, "state": "failed", "reason": str(exc)})
                if len(scores) == inner_count and admissible:
                    candidates.append((arm, float(np.mean(scores)), float(np.mean(secondary))))
            selected = (min if group == "model_1" else max)(candidates, key=lambda v: (v[1],v[2]))[0] if candidates else None
            for label, arm in (("reference", "reference"), ("selected_inner", selected)):
                try:
                    if arm is None:
                        raise ValueError("No complete inner arm")
                    result, pred, truth = evaluate(group, algorithm, arm, device, 42+fold, data, tr, te, cfg)
                    provenance.extend({"algorithm": algorithm, "fold": fold, "arm": arm, "stage": "outer", **p} for p in result.pop("generated"))
                    results.append({"algorithm": algorithm, "fold": fold, "mode": label, "arm": arm, "state": "ok", **result})
                    predictions.extend({"algorithm": algorithm, "fold": fold, "mode": label, "row_id": str(x.index[j]),
                                        "actual": float(v), "score": float(p)} for j,v,p in zip(te,truth,pred,strict=True))
                    print(f"{group} {algorithm} fold {fold} {label}/{arm}: {metric}={result[metric]:.4f}", flush=True)
                except Exception as exc:
                    results.append({"algorithm": algorithm, "fold": fold, "mode": label, "arm": arm,
                                    "state": "failed", "reason": str(exc)})
            # Persist progress after each outer fold; never declare partial runs complete.
            for name, rows in (("outer", results), ("inner", inner_results), ("predictions", predictions), ("augmentation", provenance)):
                pd.DataFrame(rows).to_csv(destination / (name+".csv"), index=False)
    table = pd.DataFrame(results)
    complete = table.loc[table.state.eq("ok")].groupby(["algorithm", "mode"]).size()
    required = len(algorithms)*2
    state = "complete" if len(complete) == required and (complete == outer_count).all() else "partial"
    if not complete.empty:
        good = table.loc[table.state.eq("ok")]
        good = good.merge(complete[complete == outer_count].rename("folds").reset_index(), on=["algorithm", "mode"])
        good.groupby(["algorithm", "mode"])[[metric, "train_"+metric]].mean().to_csv(destination / "summary.csv")
    (destination / "manifest.json").write_text(json.dumps({"state": state, "scope": "development_only",
        "prepared_identity": directory.name, "treatment_plan_identity": plan_manifest["identity"],
        "algorithms": algorithms, "arms": arms, "inner_folds": inner_count,
        "outer_folds": outer_count, "independent_holdout": False, "promotion": False,
        "reference": "same fixed default family settings; not archived Optuna winner"}, indent=2), encoding="utf-8")
    print(f"Estado experimental: {state} | {destination}")
    return destination


def main():
    from lithiumscope.datasets.management.diagnostic_provisioning import prepare_diagnostic_source
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["model_1", "model_2", "both"], default="both")
    parser.add_argument("--all", action="store_true", help="Evaluate all six families per selected model")
    args = parser.parse_args()
    for group in ("model_1", "model_2") if args.model == "both" else (args.model,):
        run_screen(group, prepare_diagnostic_source(group), all_families=args.all)


if __name__ == "__main__":
    main()
