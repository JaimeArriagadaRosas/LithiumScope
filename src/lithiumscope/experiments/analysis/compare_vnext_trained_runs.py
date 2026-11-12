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


from lithiumscope.experiments.analysis.compare_vnext_trained_runs_compare_m1 import compare_m1


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


from lithiumscope.experiments.analysis.compare_vnext_trained_runs_compare_m2 import compare_m2


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
