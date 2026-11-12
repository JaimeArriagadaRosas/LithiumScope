"""M2 experimental grouped CV, regularization, image augmentation, calibration.

Runs only against existing cached Sentinel-2 images and training data. Does not
replace active or candidate release bundles. No random fold fallback.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, brier_score_loss,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from lithiumscope.core.paths import PROJECT_ROOT, RESULTS_DIR
from lithiumscope.model_2.steps.step_01_load_imagery import load_imagery
from lithiumscope.model_2.steps.step_04_image_preprocessing import preprocess_image
from lithiumscope.model_2.steps.step_05_spectral_features import extract_spectral_features
from lithiumscope.model_2.training.scientific_validation import make_scientific_folds


def augment_image_training_only(
    cube: np.ndarray, *, seed: int = 42, brightness: float = 1.04,
    noise_fraction: float = 0.02, missing_fraction: float = 0.03,
) -> np.ndarray:
    """Small, bounded pixel-domain perturbation, not a substitute for SCL masks."""
    if not 0.0 < brightness <= 1.25:
        raise ValueError("Unexpected brightness factor")
    if not 0 <= noise_fraction <= 0.1 or not 0 <= missing_fraction <= 0.1:
        raise ValueError("Invalid augmentation intensity")
    original = np.asarray(cube, dtype=np.float32)
    if original.ndim != 3 or original.shape[0] != 6:
        raise ValueError("Expected 6-band Sentinel-2 patch")
    rng = np.random.default_rng(seed)
    value = original.copy()
    std = np.nanstd(value, axis=(1, 2), keepdims=True)
    scale = np.maximum(std, 0.005) * noise_fraction
    value = (value * brightness + rng.normal(
        0, 1, size=value.shape
    ).astype(np.float32) * scale).astype(np.float32)
    value = np.clip(value, 0, 1.6)
    value[rng.random(value.shape) < missing_fraction] = np.nan
    return value


def expected_calibration_error(
    labels, probabilities, *, bins: int = 10,
) -> float:
    labels = np.asarray(labels, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    if labels.shape != probabilities.shape or not len(labels):
        raise ValueError("Invalid labels / probabilities")
    if not np.isfinite(probabilities).all():
        raise ValueError("Probabilities contain non-finite values")
    edges = np.linspace(0, 1, bins+1)
    result = 0.0
    for idx in range(bins):
        mask = ((probabilities >= edges[idx])
                & (probabilities < edges[idx+1]
                   if idx < bins - 1 else probabilities <= edges[idx+1]))
        if mask.any():
            result += float(mask.mean()) * abs(
                float(labels[mask].mean()) - float(probabilities[mask].mean())
            )
    return float(result)


def score_metrics(y, p) -> dict[str, float]:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    if not np.isin(y, [0, 1]).all() or len(np.unique(y)) < 2:
        raise ValueError("Heldout fold must contain both classes")
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "balanced_accuracy": float(balanced_accuracy_score(y, p>=0.5)),
        "f1": float(f1_score(y, p>=0.5, zero_division=0)),
        "precision": float(precision_score(y, p>=0.5, zero_division=0)),
        "recall": float(recall_score(y, p>=0.5, zero_division=0)),
        "brier": float(brier_score_loss(y,p)),
        "ece_10": expected_calibration_error(y,p),
    }


def _calibration_splits(x, y, groups, n_splits: int = 3):
    if groups.nunique() < n_splits:
        return None
    splits = list(GroupKFold(n_splits=n_splits).split(x, groups=groups))
    for tr, va in splits:
        if len(np.unique(np.asarray(y)[tr])) < 2 or len(np.unique(np.asarray(y)[va])) < 2:
            return None
        if set(groups.iloc[tr]).intersection(set(groups.iloc[va])):
            raise RuntimeError("Group leakage during probability calibration")
    return splits


def make_augmentation_table(
    manifest: pd.DataFrame,
    bundle: dict,
    *,
    seed: int = 42,
) -> pd.DataFrame:
    features = list(bundle["features"])
    names = tuple(bundle["band_names"])
    rows = []
    for idx, item in enumerate(manifest.itertuples(index=False)):
        image_path = Path(item.image_path)
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        image = load_imagery(image_path)
        if image.shape[0] != len(names):
            raise ValueError("Sentinel band mismatch: "+str(image_path))
        disturbed = augment_image_training_only(image, seed=seed+idx)
        clean = preprocess_image(
            disturbed, normalize_per_band=bool(bundle.get("normalize_per_band",False))
        )
        result = extract_spectral_features(clean, band_names=names)
        rows.append(result)
        if (idx+1)%100==0:
            print("M2_AUG_FEATURES",idx+1,"/",len(manifest),flush=True)
    table=pd.DataFrame(rows).reindex(columns=features)
    if table.isna().any().any() or not np.isfinite(table.to_numpy(dtype=float)).all():
        raise RuntimeError("Augmented training features are not all finite.")
    return table


from lithiumscope.experiments.analysis.run_vnext_m2_run_experiments import run_experiments


if __name__=="__main__":
    result=run_experiments()
    print("M2_DEVELOPMENT_DONE",result["selected"],flush=True)
