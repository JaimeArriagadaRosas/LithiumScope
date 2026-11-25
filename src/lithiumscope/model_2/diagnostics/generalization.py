"""Reproducible overfit/underfit screen for a frozen M2 candidate.

This screen compares a final estimator's in-sample fit with previously saved
outer-fold OOF scores. Different fitting population and fold-specific labels
mean it is NOT the definitive nested, paired train-vs-test gap.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    roc_auc_score,
)


def _metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    valid = np.isfinite(scores)
    y = labels[valid]
    probability = scores[valid]
    if len(y) < 2 or len(np.unique(y)) != 2:
        raise ValueError("Cannot compare without two classes and finite scores")
    return {
        "n": int(len(y)),
        "positives": int(np.sum(y)),
        "roc_auc": float(roc_auc_score(y, probability)),
        "average_precision": float(average_precision_score(y, probability)),
        "brier_score": float(brier_score_loss(y, probability)),
        "balanced_accuracy_0p5": float(
            balanced_accuracy_score(y, probability >= 0.5)
        ),
    }


def audit_candidate(
    bundle_path: Path,
    spectral_features_path: Path,
    quality_audit_path: Path,
    outer_oof_path: Path,
) -> dict:
    """Only loads immutable artifacts; does not retrain or change candidates."""
    bundle = joblib.load(bundle_path)
    features = pd.read_csv(spectral_features_path)
    audit = pd.read_csv(quality_audit_path)
    oof = pd.read_csv(outer_oof_path)
    needed = list(bundle["features"])
    if len(features) != len(audit) or len(features) != len(oof):
        raise ValueError("Features, spectral quality and OOF must align")
    if not set(needed).issubset(features):
        raise ValueError("Spectral cache missing fitted feature columns")
    if not {"train_quality_pass_95", "Li_icpms"}.issubset(audit):
        raise ValueError("Required quality audit columns absent")
    if not {"li_ppm", "prospectivity_score", "y_true"}.issubset(oof):
        raise ValueError("Required OOF columns absent")
    li = features["Li_icpms"].to_numpy(float)
    if not np.allclose(li, audit.Li_icpms.to_numpy(float)):
        raise ValueError("Audit Li labels do not match cached features")
    if not np.allclose(li, oof.li_ppm.to_numpy(float)):
        raise ValueError("OOF rows reordered relative to final training features")

    keep = audit.train_quality_pass_95.to_numpy(dtype=bool)
    threshold = float(bundle["threshold_ppm"])
    fixed_labels = (li >= threshold).astype(int)
    train_predictions = np.asarray(
        bundle["estimator"].predict_proba(features.loc[keep, needed])[:, 1],
        dtype=float,
    )
    outer_predictions = oof.prospectivity_score.to_numpy(float)[keep]
    fit = _metrics(fixed_labels[keep], train_predictions)
    hold = _metrics(fixed_labels[keep], outer_predictions)
    fold_labels = _metrics(oof.y_true.to_numpy(dtype=int)[keep], outer_predictions)
    return {
        "candidate_path": str(bundle_path),
        "candidate_algorithm": str(bundle["algorithm"]),
        "threshold_ppm": threshold,
        "features": len(needed),
        "train_eligible_rows": int(np.sum(keep)),
        "final_model_apparent_training": fit,
        "spatial_outer_oof_same_rows_fixed_lithium_threshold": hold,
        "spatial_outer_oof_fold_specific_labels": fold_labels,
        "apparent_roc_auc_gap": fit["roc_auc"] - hold["roc_auc"],
        "apparent_ap_gap": fit["average_precision"] - hold["average_precision"],
        "strong_overfit_warning": (fit["roc_auc"] - hold["roc_auc"]) > 0.15,
        "underfit_supported": (
            fit["roc_auc"] <= 0.6 and hold["roc_auc"] <= 0.6
        ),
        "not_a_release_gate": True,
        "limitations": [
            "The final in-sample estimator is fitted on all QA-eligible data.",
            "Outer OOF estimates were fitted on spatially held-out folds.",
            "OOF labels were originally fold-dependent; fixed-target scoring "
            "is a sensitivity analysis, not an identical training experiment.",
            "Require matched per-fold train/test evaluation before declaring "
            "the size of overfitting or selecting regularization.",
        ],
    }
