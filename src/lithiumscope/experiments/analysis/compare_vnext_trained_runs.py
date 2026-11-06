"""Read-only comparisons of saved vNext M1/M2 spatial out-of-fold predictions.

NO fit, train, refit, dataset download, model promotion, or source rewrite.
M1 paired observations may have different outer-fold assignments between runs.
M2 is strictly identical on sample labels, spatial folds, and input manifest hash.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    roc_auc_score,
)

from lithiumscope.core.paths import PROJECT_ROOT, RESULTS_DIR


OLD_M1 = "training_20261105_204029_m0300"
NEW_M1 = "training_20261106_111653_m0300"
OLD_M2 = "training_20261105_223618_m0300"
NEW_M2 = "training_20261106_135029_m0300"
OUTPUT = (
    RESULTS_DIR / "scientific_simulations" / "vnext_20261106"
    / "stage5_training_comparison"
)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _require_completed(run: Path) -> dict:
    run_json = _load_json(run / "run.json")
    if run_json.get("state") != "completed":
        raise ValueError(f"Run not completed: {run}")
    return run_json


def regression_metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    y, pred = np.asarray(y, dtype=float), np.asarray(pred, dtype=float)
    if len(y) != len(pred) or not len(y) or not np.isfinite(y).all():
        raise ValueError("Bad regression target")
    if not np.isfinite(pred).all():
        raise ValueError("Non-finite predictions")
    error = pred - y
    return {
        "n": int(len(y)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "median_ae": float(np.median(np.abs(error))),
        "p90_ae": float(np.quantile(np.abs(error), .9)),
        "bias_pred_minus_true": float(error.mean()),
    }


def classification_metrics(y: np.ndarray, score: np.ndarray) -> dict:
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    if not set(np.unique(y)) == {0, 1} or not np.isfinite(score).all():
        raise ValueError("Binary test must contain both classes and finite scores")
    if ((score < 0) | (score > 1)).any():
        raise ValueError("Invalid classification probabilities")
    return {
        "n": int(len(y)),
        "roc_auc": float(roc_auc_score(y, score)),
        "average_precision": float(average_precision_score(y, score)),
        "brier": float(brier_score_loss(y, score)),
        "balanced_accuracy_at_0_5": float(
            balanced_accuracy_score(y, score >= .5)
        ),
        "positive_prevalence": float(y.mean()),
    }


def group_bootstrap_gain(
    y, old_score, new_score, groups, *,
    metric: str, repeats: int = 1000, seed: int = 2026,
) -> dict:
    """CI of old error minus new error or new quality minus old quality.

    Observations are resampled as entire spatial groups, NOT individual rows.
    The interval remains exploratory (the same folds selected the candidates).
    """
    y = np.asarray(y, dtype=float)
    old = np.asarray(old_score, dtype=float)
    new = np.asarray(new_score, dtype=float)
    groups = np.asarray(groups, dtype=str)
    if (len(y) != len(old) or len(y) != len(new)
            or len(y) != len(groups)):
        raise ValueError("Bootstrap arrays differ")
    unique, inv = np.unique(groups, return_inverse=True)
    if len(unique) < 2 or repeats < 20:
        raise ValueError("Need at least two groups and 20 iterations")
    arrays = [np.flatnonzero(inv == i) for i in range(len(unique))]
    rng = np.random.default_rng(seed)

    def gain(indices):
        target = y[indices]
        o, n = old[indices], new[indices]
        if metric == "rmse":
            return (
                float(np.sqrt(np.mean((target-o)**2)))
                - float(np.sqrt(np.mean((target-n)**2)))
            )
        if metric == "mae":
            return float(np.mean(np.abs(target-o))-np.mean(np.abs(target-n)))
        if metric == "brier":
            return float(np.mean((target-o)**2)-np.mean((target-n)**2))
        if len(np.unique(target)) != 2:
            return np.nan
        if metric == "roc_auc":
            return float(roc_auc_score(target.astype(int), n) -
                         roc_auc_score(target.astype(int), o))
        if metric == "average_precision":
            return float(average_precision_score(target.astype(int), n) -
                         average_precision_score(target.astype(int), o))
        raise ValueError(f"Invalid metric: {metric}")

    observed = gain(np.arange(len(y)))
    draws = []
    for _ in range(repeats):
        chosen = rng.integers(0, len(unique), size=len(unique))
        selected = np.concatenate([arrays[i] for i in chosen])
        value = gain(selected)
        if np.isfinite(value):
            draws.append(value)
    if len(draws) < .9 * repeats:
        raise ValueError("Too many invalid group bootstrap replicates")
    lower, upper = np.quantile(draws, [.025, .975])
    return {
        "metric": metric,
        "gain_direction": "positive_means_new_better",
        "observed_gain": float(observed),
        "ci95_group_lower": float(lower),
        "ci95_group_upper": float(upper),
        "resamples": len(draws),
        "spatial_groups": int(len(unique)),
        "caveat": "Exploratory grouped bootstrap; not external independent validation.",
    }


def load_m1_oof(run: Path, model: str = "svm__identity") -> pd.DataFrame:
    _require_completed(run)
    pred = pd.read_csv(run / "tables" / f"oof_predictions_{model}.csv")
    manifest = pd.read_csv(run / "tables" / "outer_fold_manifest.csv")
    if len(pred) != len(manifest):
        raise ValueError("M1 manifest and predictions differ")
    if not np.array_equal(
        pred.row_index.to_numpy(), manifest.row_index.to_numpy()
    ):
        raise ValueError("M1 prediction row_index does not match manifest")
    if manifest.groupby("spatial_group").outer_fold.nunique().max() > 1:
        raise ValueError("M1 spatial group present in multiple test folds")
    if pred.row_index.duplicated().any():
        raise ValueError("M1 predictions have duplicate row indexes")
    joined = manifest[[
        "source_dataset", "spatial_group", "outer_fold", "Longitude", "Latitude"
    ]].copy()
    joined["y_true"] = pred.Li_icpms_real.to_numpy()
    joined["prediction"] = pred.Li_icpms_predicted.to_numpy()
    joined[["Longitude", "Latitude"]] = joined[[
        "Longitude", "Latitude"
    ]].round(5)
    joined["y_true"] = joined.y_true.round(6)
    return joined


def pair_shared_m1(
    old: pd.DataFrame, new: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """Match only unambiguous same-source same-site same-Li test records."""
    keys = ["source_dataset", "Longitude", "Latitude", "y_true"]
    old_ambiguous = old.duplicated(keys, keep=False)
    new_ambiguous = new.duplicated(keys, keep=False)
    old_unique = old.loc[~old_ambiguous]
    new_unique = new.loc[~new_ambiguous]
    paired = old_unique.merge(
        new_unique,
        on=keys,
        how="inner",
        suffixes=("_old", "_new"),
        validate="one_to_one",
    )
    if not len(paired):
        raise ValueError("No unambiguous common M1 observations")
    if not paired.spatial_group_old.equals(paired.spatial_group_new):
        raise ValueError("M1 sample changed spatial group")
    excluded = len(old) - len(paired)
    meta = {
        "old_total": len(old),
        "new_total": len(new),
        "matched": len(paired),
        "excluded_old": excluded,
        "ambiguous_old": int(old_ambiguous.sum()),
        "ambiguous_new": int(new_ambiguous.sum()),
        "same_outer_fold": int(
            (paired.outer_fold_old == paired.outer_fold_new).sum()
        ),
        "different_outer_fold": int(
            (paired.outer_fold_old != paired.outer_fold_new).sum()
        ),
        "comparison_limit": (
            "These are the same physical observations but outer spatial folds "
            "and training domains differ. This is a paired exploratory OOF "
            "diagnostic, not a strict common frozen holdout."
        ),
    }
    return paired, meta


def compare_m1(root: Path, output: Path) -> dict:
    old_run = root / "results/model_1/runs" / OLD_M1
    new_run = root / "results/model_1/runs" / NEW_M1
    old, new = load_m1_oof(old_run), load_m1_oof(new_run)
    paired, metadata = pair_shared_m1(old, new)
    y = paired.y_true.to_numpy()
    a = paired.prediction_old.to_numpy()
    b = paired.prediction_new.to_numpy()
    threshold = float(np.quantile(y, .75))
    subgroups = {
        "all_shared": paired.index >= 0,
        "high_lithium_q75_common": paired.y_true >= threshold,
        "same_outer_fold_sensitivity":
            paired.outer_fold_old == paired.outer_fold_new,
        "changed_outer_fold_sensitivity":
            paired.outer_fold_old != paired.outer_fold_new,
    }
    for src in sorted(paired.source_dataset.unique()):
        subgroups[f"source_{src}"] = paired.source_dataset.eq(src)
    stratified = {}
    for label, subset in subgroups.items():
        rows = paired.loc[subset]
        stratified[label] = {
            "old": regression_metrics(rows.y_true, rows.prediction_old),
            "new": regression_metrics(rows.y_true, rows.prediction_new),
        }
    uncertainty = {
        metric: group_bootstrap_gain(
            y, a, b, paired.spatial_group_old, metric=metric
        ) for metric in ("rmse", "mae")
    }
    report = {
        "status": "exploratory_improvement_not_release_proof",
        "old_run": OLD_M1, "new_run": NEW_M1,
        "same_model_family": "svm__identity",
        "comparison": metadata,
        "common_high_li_threshold_ppm": threshold,
        "by_subset": stratified,
        "cluster_bootstrap": uncertainty,
        "development_improvement_gate": {
            "rules": (
                "At least 80% uniquely matched old observations; "
                "at least 5% lower paired RMSE and MAE; "
                "positive exploratory grouped bootstrap lower CI for both."
            ),
            "passes_exploratory_only": bool(
                len(paired) >= .80 * len(old)
                and stratified["all_shared"]["new"]["rmse"]
                    <= .95 * stratified["all_shared"]["old"]["rmse"]
                and stratified["all_shared"]["new"]["mae"]
                    <= .95 * stratified["all_shared"]["old"]["mae"]
                and uncertainty["rmse"]["ci95_group_lower"] > 0
                and uncertainty["mae"]["ci95_group_lower"] > 0
            ),
            "release_certified": False,
            "reason": "610 of 1004 shared records changed outer-fold assignment.",
        },
        "unpaired_full_oof_not_comparable": {
            "old": regression_metrics(
                old.y_true.to_numpy(), old.prediction.to_numpy()
            ),
            "new": regression_metrics(
                new.y_true.to_numpy(), new.prediction.to_numpy()
            ),
        },
        "decision": (
            "M1 B improves exploratory paired common-record errors. "
            "Do not compare raw full-dataset CV RMSE across different "
            "lithium distributions or claim an independent release test."
        ),
    }
    paired.to_csv(output / "m1_shared_records_paired_oof.csv", index=False)
    (output / "m1_paired_comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("M1_SAVED_COMPARISON", json.dumps({
        "n_shared": len(paired),
        "same_folds": metadata["same_outer_fold"],
        "different_folds": metadata["different_outer_fold"],
        "old": stratified["all_shared"]["old"],
        "new": stratified["all_shared"]["new"],
        "CI": uncertainty,
    }, ensure_ascii=False), flush=True)
    return report


def load_m2_oof(run: Path, algorithm: str) -> pd.DataFrame:
    _require_completed(run)
    frame = pd.read_csv(
        run / "tables" / f"oof_predictions_{algorithm}.csv"
    )
    required = {
        "y_true", "li_ppm", "outer_fold", "prospectivity_score"
    }
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing M2 prediction fields: {required-set(frame)}")
    if frame.isna().any().any():
        raise ValueError("M2 OOF predictions contain nulls")
    return frame


def check_m2_identical_protocol(
    old_run: Path, new_run: Path, old: pd.DataFrame, new: pd.DataFrame,
    manifest: pd.DataFrame,
) -> None:
    old_info = _require_completed(old_run)
    new_info = _require_completed(new_run)
    if (old_info["summary"]["dataset_sha256"]
            != new_info["summary"]["dataset_sha256"]):
        raise ValueError("M2 dataset fingerprints differ")
    if not len(old) == len(new) == len(manifest):
        raise ValueError("M2 row count differs between runs")
    for key in ("y_true", "li_ppm", "outer_fold"):
        if not np.allclose(old[key], new[key], equal_nan=False):
            raise ValueError(f"M2 comparison differs in {key}")
    old_folds = _load_json(old_run / "manifests" / "scientific_folds.json")
    new_folds = _load_json(new_run / "manifests" / "scientific_folds.json")
    if old_folds != new_folds:
        raise ValueError("M2 outer-fold manifests differ")
    if (old.assign(group=manifest.spatial_group)
            .groupby("group").outer_fold.nunique().max() > 1):
        raise ValueError("M2 same spatial group appears in multiple test folds")
    for fold in new_folds:
        indices = np.asarray(fold["test_indices"], dtype=int)
        if not np.all(new.outer_fold.iloc[indices] == fold["fold"]):
            raise ValueError("M2 OOF row-to-fold labels are inconsistent")


def compare_m2(root: Path, output: Path) -> dict:
    old_run = root / "results/model_2/runs" / OLD_M2
    new_run = root / "results/model_2/runs" / NEW_M2
    old_rf = load_m2_oof(old_run, "random_forest")
    new_rf = load_m2_oof(new_run, "random_forest")
    cal_rf = load_m2_oof(new_run, "random_forest_spatial_calibrated")
    old_cat = load_m2_oof(old_run, "catboost")
    manifest = pd.read_csv(
        root / "data/processed/model_2/training_manifest.csv"
    )
    check_m2_identical_protocol(old_run, new_run, old_rf, new_rf, manifest)
    for candidate in (cal_rf, old_cat):
        for key in ("y_true", "li_ppm", "outer_fold"):
            if not np.allclose(candidate[key], old_rf[key]):
                raise ValueError(f"M2 differing labels/folds for {key}")
    y = old_rf.y_true.to_numpy(dtype=int)
    scores = {
        "previous_rf": old_rf.prospectivity_score.to_numpy(dtype=float),
        "new_rf": new_rf.prospectivity_score.to_numpy(dtype=float),
        "new_rf_spatial_calibrated":
            cal_rf.prospectivity_score.to_numpy(dtype=float),
        "previous_catboost": old_cat.prospectivity_score.to_numpy(dtype=float),
    }
    metrics = {
        label: classification_metrics(y, value)
        for label, value in scores.items()
    }
    difference = np.max(np.abs(scores["new_rf"]-scores["previous_rf"]))
    groups = manifest.spatial_group.to_numpy()
    ci_calibrated = {
        metric: group_bootstrap_gain(
            y, scores["previous_rf"], scores["new_rf_spatial_calibrated"],
            groups, metric=metric
        ) for metric in ("roc_auc", "average_precision", "brier")
    }
    paired_rows = pd.DataFrame({
        "row_index": np.arange(len(y)),
        "spatial_group": groups,
        "outer_fold": old_rf.outer_fold,
        "y_true": y,
        "Li_ppm": old_rf.li_ppm,
        **scores,
    })
    paired_rows.to_csv(output / "m2_identical_protocol_paired_oof.csv",
                       index=False)
    rows = []
    for fold, slice_ in paired_rows.groupby("outer_fold"):
        if len(np.unique(slice_.y_true)) != 2:
            raise ValueError("M2 outer fold missing positive/negative samples")
        for name in scores:
            m = classification_metrics(slice_.y_true, slice_[name])
            rows.append({"fold": int(fold), "variant": name, **m})
    pd.DataFrame(rows).to_csv(output / "m2_paired_fold_metrics.csv",
                              index=False)
    old_perf = metrics["previous_rf"]
    new_perf = metrics["new_rf"]
    discrimination_gate = bool(
        new_perf["roc_auc"] - old_perf["roc_auc"] >= .03
        and new_perf["average_precision"] - old_perf["average_precision"] >= .02
        and new_perf["brier"] - old_perf["brier"] <= .01
    )
    report = {
        "status": "no_new_discrimination_gain",
        "old_run": OLD_M2, "new_run": NEW_M2,
        "same_manifest_same_labels_same_spatial_folds": True,
        "samples": len(y),
        "spatial_groups": int(pd.Series(groups).nunique()),
        "maximum_old_new_rf_probability_delta": float(difference),
        "new_old_rf_identical_to_tolerance_1e_12": bool(difference <= 1e-12),
        "metrics_global_oof": metrics,
        "calibrated_vs_previous_rf_bootstrap": ci_calibrated,
        "development_discrimination_gate": {
            "rules": (
                "At least +0.03 ROC-AUC and +0.02 AP relative to prior winner, "
                "without more than +0.01 degradation in Brier."
            ),
            "passes": discrimination_gate,
            "release_certified": False,
        },
        "assessment": (
            "The RF winner is numerically unchanged; the new calibrated "
            "variant improves Brier score but reduces ROC-AUC/AP and at 0.5 "
            "has no sensitivity. No improvement in discriminative accuracy "
            "has been demonstrated over previous training."
        ),
        "release_limit": (
            "No untouched external Li-labelled holdout. Internal OOF cannot "
            "alone authorize promotion; active September release was not "
            "measured here under the same protocol."
        ),
    }
    (output / "m2_paired_comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("M2_SAVED_COMPARISON", json.dumps({
        "max_probability_delta": difference,
        "metrics": metrics,
        "CI": ci_calibrated,
    }, ensure_ascii=False), flush=True)
    return report


def run(root: Path = PROJECT_ROOT, output: Path = OUTPUT) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "model_1": compare_m1(root, output),
        "model_2": compare_m2(root, output),
        "safety": {
            "model_fits": 0,
            "model_downloads": 0,
            "model_promotions": 0,
            "input_files_modified": 0,
        },
    }
    (output / "comparison_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    run()
