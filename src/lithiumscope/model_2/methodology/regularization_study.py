"""Controlled RF spectral overfit study; never promotes or overwrites a model.

Fixed original 706 historical folds + anchored GEOROC groups. The inner folds
belong strictly to outer training and are used for threshold/calibration.
All outer test rows (including QA-failed) remain untouched. No 10-case demo
is used for tuning or promotion.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold

from lithiumscope.core.paths import PROJECT_ROOT
from lithiumscope.model_2.evaluation.thresholds import select_operating_threshold
from lithiumscope.model_2.training.scientific_validation import (
    make_anchored_scientific_folds,
)


def _metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    y = np.asarray(labels, dtype=int)
    p = np.asarray(scores, dtype=float)
    if len(np.unique(y)) != 2:
        raise ValueError("Cannot score fold with a missing class")
    return {
        "n": len(y),
        "positives": int(y.sum()),
        "roc_auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "balanced_accuracy": float(balanced_accuracy_score(y, p >= threshold)),
        "threshold": float(threshold),
        "predicted_positive": int(np.sum(p >= threshold)),
    }


def _calibrator(scores: np.ndarray, labels: np.ndarray):
    """Platt sigmoid fit on out-of-fold inner-training predictions only."""
    x = np.log(np.clip(scores, 1e-5, 1 - 1e-5) /
               (1 - np.clip(scores, 1e-5, 1 - 1e-5))).reshape(-1, 1)
    model = LogisticRegression(C=1.0, max_iter=500, random_state=42)
    model.fit(x, labels)
    return model


def _apply_calibrator(calibrator, scores: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(scores, dtype=float), 1e-5, 1 - 1e-5)
    return calibrator.predict_proba(np.log(p / (1 - p)).reshape(-1, 1))[:, 1]


def _inner_scores(estimator, frame, labels, groups, seed: int) -> np.ndarray:
    splitter = GroupKFold(n_splits=3)
    score = np.full(len(frame), np.nan)
    splits = list(splitter.split(frame, labels, groups))
    for k, (train_idx, valid_idx) in enumerate(splits):
        if set(groups.iloc[train_idx]) & set(groups.iloc[valid_idx]):
            raise ValueError("Inner spatial leakage")
        if len(np.unique(labels[train_idx])) != 2:
            raise ValueError("Inner train missing a lithium class")
        model = clone(estimator).set_params(random_state=seed + k)
        model.fit(frame.iloc[train_idx], labels[train_idx])
        score[valid_idx] = model.predict_proba(frame.iloc[valid_idx])[:, 1]
    if not np.isfinite(score).all():
        raise ValueError("Incomplete inner out-of-fold probabilities")
    return score


def _load_inputs(root: Path):
    manifest = pd.read_csv(
        root / "data/processed/model_2/vnext_expanded/training_manifest.csv"
    )
    features = pd.read_csv(
        root / "data/processed/model_2/vnext_expanded/spectral_features.csv"
    )
    audit = pd.read_csv(
        root / "results/model_2/runs/training_20261112_123451_m0300"
        / "tables/scientific_sample_audit_783.csv"
    )
    legacy = json.loads(
        (root / "config/model_2_frozen_spatial_folds_706.json").read_text(
            encoding="utf-8"
        )
    )
    bundle = joblib.load(
        root / "models/model_2/trained/training_20261112_123451_m0300/model.joblib"
    )
    if not (len(manifest) == len(features) == len(audit) == 783):
        raise ValueError("Dataset drift: expected the verified 783-row snapshot")
    if not np.array_equal(manifest.sample_id.astype(str), audit.sample_id.astype(str)):
        raise ValueError("QA rows are not aligned with source samples")
    if not np.array_equal(
        manifest.spatial_group.astype(str), features.spatial_group.astype(str)
    ):
        raise ValueError("Cached features have inconsistent spatial groups")
    if not np.allclose(manifest.Li_icpms, features.Li_icpms):
        raise ValueError("Geochemistry cache differs from training manifest")
    if not set(bundle["features"]).issubset(features):
        raise ValueError("Missing spectral features")
    folds = make_anchored_scientific_folds(
        manifest.Li_icpms, manifest.spatial_group, legacy, baseline_count=706,
        quantile=0.75,
    )
    return manifest, features[bundle["features"]], audit, bundle, folds


def run_study(*, root: Path = PROJECT_ROOT, output: Path) -> dict:
    manifest, x, audit, bundle, folds = _load_inputs(root)
    eligible = audit.train_quality_pass_95.to_numpy(dtype=bool)
    groups = manifest.spatial_group.astype(str)
    # One controlled change: regularize tree depth and minimum leaf population.
    # Other RF settings (feature sampling, class weights, trees, seed) stay fixed.
    arms = {
        "frozen_rf_configuration": {},
        "regularized_depth8_leaf10": {
            "max_depth": 8,
            "min_samples_leaf": 10,
        },
    }
    rows, folds_report = [], []
    for number, fold in enumerate(folds, start=1):
        train_idx = fold.train_idx[eligible[fold.train_idx]]
        test_idx = fold.test_idx
        if set(groups.iloc[train_idx]) & set(groups.iloc[test_idx]):
            raise ValueError("Outer spatial overlap")
        train_y = (
            manifest.Li_icpms.iloc[train_idx].to_numpy(float)
            >= fold.label_threshold_ppm
        ).astype(int)
        test_y = np.asarray(fold.y_test, dtype=int)
        if len(np.unique(train_y)) != 2 or len(np.unique(test_y)) != 2:
            raise ValueError("Missing outer train/test class")
        train_x, test_x = x.iloc[train_idx], x.iloc[test_idx]
        for arm, overrides in arms.items():
            estimator = clone(bundle["estimator"]).set_params(**overrides)
            inner = _inner_scores(
                estimator, train_x.reset_index(drop=True), train_y,
                groups.iloc[train_idx].reset_index(drop=True), seed=42 + number * 10,
            )
            calibrated = _calibrator(inner, train_y)
            estimator.fit(train_x, train_y)
            outer = estimator.predict_proba(test_x)[:, 1]
            apparent = estimator.predict_proba(train_x)[:, 1]
            for method in ("raw", "inner_sigmoid"):
                if method == "raw":
                    inner_use, outer_use, apparent_use = inner, outer, apparent
                else:
                    inner_use = _apply_calibrator(calibrated, inner)
                    outer_use = _apply_calibrator(calibrated, outer)
                    apparent_use = _apply_calibrator(calibrated, apparent)
                choice = select_operating_threshold(
                    train_y, inner_use, objective="balanced_accuracy"
                )
                test_metrics = _metrics(test_y, outer_use, choice.threshold)
                fit_metrics = _metrics(train_y, apparent_use, choice.threshold)
                folds_report.append({
                    "fold": number, "arm": arm, "calibration": method,
                    "train_rows": len(train_idx), "test_rows": len(test_idx),
                    "train_groups": groups.iloc[train_idx].nunique(),
                    "test_groups": groups.iloc[test_idx].nunique(),
                    "target_li_ppm": fold.label_threshold_ppm,
                    "apparent_train_auc": fit_metrics["roc_auc"],
                    "outer_test_auc": test_metrics["roc_auc"],
                    "generalization_auc_gap": (
                        fit_metrics["roc_auc"] - test_metrics["roc_auc"]
                    ),
                    **{f"test_{k}": v for k, v in test_metrics.items()},
                })
                for pos, original_idx in enumerate(test_idx):
                    rows.append({
                        "sample_id": manifest.sample_id.iloc[original_idx],
                        "sample_index": int(original_idx),
                        "spatial_group": groups.iloc[original_idx],
                        "source_dataset": manifest.source_dataset.iloc[original_idx],
                        "Li_icpms": float(manifest.Li_icpms.iloc[original_idx]),
                        "fold": number, "arm": arm, "calibration": method,
                        "y_true": int(test_y[pos]),
                        "score": float(outer_use[pos]),
                        "threshold_from_inner": float(choice.threshold),
                        "predicted_positive": bool(
                            outer_use[pos] >= choice.threshold
                        ),
                    })
        print(f"Completed untouched spatial outer fold {number}/5", flush=True)

    output.mkdir(parents=True, exist_ok=True)
    per_case = pd.DataFrame(rows)
    per_fold = pd.DataFrame(folds_report)
    per_case.to_csv(output / "paired_outer_predictions.csv", index=False)
    per_fold.to_csv(output / "fold_metrics.csv", index=False)
    summary = {}
    for (arm, method), subset in per_case.groupby(["arm", "calibration"]):
        scores = subset.score.to_numpy(float)
        target = subset.y_true.to_numpy(int)
        classed = subset.predicted_positive.to_numpy(bool)
        summary[f"{arm}/{method}"] = {
            "n": len(subset),
            "roc_auc": float(roc_auc_score(target, scores)),
            "average_precision": float(average_precision_score(target, scores)),
            "brier": float(brier_score_loss(target, scores)),
            "balanced_accuracy_nested_threshold": float(
                balanced_accuracy_score(target, classed)
            ),
            "positive_predictions": int(classed.sum()),
            "mean_apparent_auc_gap": float(
                per_fold.loc[
                    (per_fold.arm == arm) & (per_fold.calibration == method),
                    "generalization_auc_gap",
                ].mean()
            ),
        }
    result = {
        "state": "exploratory_completed_not_release_approved",
        "n_samples": 783, "train_qa95_eligible": int(eligible.sum()),
        "historical_folds_frozen": True,
        "test_rows_preserved": 783,
        "external_ten_case_demo_used_for_selection": False,
        "regularization_technique": "RandomForest max_depth 8 / min_samples_leaf 10",
        "method": "paired_outer_spatial_CV_inner_group_CV_sigmoid_and_threshold",
        "summary": summary,
        "limitations": (
            "Candidate algorithms and thresholds are compared on reusable "
            "development folds. Release requires a genuinely independent "
            "external geographical and source holdout with uncertainty."
        ),
    }
    (output / "experiment_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_study(output=args.output), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
