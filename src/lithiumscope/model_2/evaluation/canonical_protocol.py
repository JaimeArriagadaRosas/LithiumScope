"""Frozen development protocol for M2 spatial generalization experiments.

This protocol is exploratory: five outer folds have been used during
development and are not an untouched confirmatory external holdout.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lithiumscope.model_2.methodology.regularization_study import _load_inputs

INPUTS = {
    "manifest": "data/processed/model_2/vnext_expanded/training_manifest.csv",
    "features": "data/processed/model_2/vnext_expanded/spectral_features.csv",
    "quality": (
        "results/model_2/runs/training_20261112_123451_m0300/"
        "tables/scientific_sample_audit_783.csv"
    ),
    "frozen_folds": "config/model_2_frozen_spatial_folds_706.json",
    "reference_model": (
        "models/model_2/trained/training_20261112_123451_m0300/model.joblib"
    ),
    "active_registry": "models/active_models.json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_folds(
    manifest: pd.DataFrame,
    eligible: np.ndarray,
    folds,
    *,
    historical_count: int = 706,
) -> list[dict]:
    n = len(manifest)
    if n != 783 or int(np.sum(eligible)) != 767:
        raise ValueError("M2 canonical source or QA95 population changed")
    if len(folds) != 5 or historical_count != 706:
        raise ValueError("Expected five frozen historical M2 folds")
    if len(eligible) != n or eligible.dtype != bool:
        raise ValueError("Missing or misaligned QA95 training mask")
    seen = []
    groups = manifest.spatial_group.astype(str)
    row_info = []
    for number, fold in enumerate(folds, 1):
        train = np.asarray(fold.train_idx, dtype=int)
        test = np.asarray(fold.test_idx, dtype=int)
        if (np.any(train < 0) or np.any(test < 0)
                or np.any(train >= n) or np.any(test >= n)):
            raise ValueError("Out-of-bounds spatial fold")
        if set(train) & set(test):
            raise ValueError("Row-level train/test overlap")
        if set(groups.iloc[train]) & set(groups.iloc[test]):
            raise ValueError("Spatial group leakage")
        train_qa = train[eligible[train]]
        if len(train_qa) < 100:
            raise ValueError("Insufficient QA-eligible training data")
        if len(set(np.asarray(fold.y_train, dtype=int))) != 2:
            raise ValueError("Missing train class")
        if len(set(np.asarray(fold.y_test, dtype=int))) != 2:
            raise ValueError("Missing test class")
        if not np.isfinite(fold.label_threshold_ppm):
            raise ValueError("Missing fold-training-only target threshold")
        seen.extend(test.tolist())
        row_info.append({
            "fold": number,
            "train_rows_before_qa95": int(len(train)),
            "train_rows_after_qa95": int(len(train_qa)),
            "test_rows_unfiltered": int(len(test)),
            "train_groups": int(groups.iloc[train_qa].nunique()),
            "test_groups": int(groups.iloc[test].nunique()),
            "test_positives": int(np.sum(fold.y_test)),
            "target_ppm_training_only": float(fold.label_threshold_ppm),
        })
    if sorted(seen) != list(range(n)):
        raise ValueError("Outer folds must cover all 783 samples exactly once")
    return row_info


def load_protocol(root: Path):
    """Return frozen inputs and their immutable, inspectable contract."""
    root = root.resolve()
    manifest, features, quality, bundle, folds = _load_inputs(root)
    eligible = quality.train_quality_pass_95.to_numpy(dtype=bool)
    if len(bundle["features"]) != 126 or len(features.columns) != 126:
        raise ValueError("Canonical protocol requires all 126 spectral features")
    if bundle["algorithm"] != "random_forest":
        raise ValueError("Reference bundle changed")
    if int((quality.source_cohort == "historical").sum()) != 706:
        raise ValueError("Expected 706 historical, 77 provisional GEOROC")
    info = validate_folds(manifest, eligible, folds)
    hashes = {name: sha256(root / path) for name, path in INPUTS.items()}
    metadata = {
        "protocol_version": "m2_spatial_rf_126_qa95_783_v1",
        "status": "development_only_not_external_confirmation",
        "historical_rows": 706,
        "provisional_georoc_rows": 77,
        "total_rows": 783,
        "train_qa95_eligible_total": 767,
        "outer_test_cases": 783,
        "features": list(bundle["features"]),
        "feature_count": 126,
        "outer_folds": info,
        "input_sha256": hashes,
        "target_policy": (
            "historical_706_q75_per_outer_training_fold; "
            "fixed within inner hyperparameter search"
        ),
        "quality_policy": "QA95 applied to train only; all outer test kept",
        "inner_policy": (
            "three GroupKFold folds within outer train; no outer test "
            "used for Optuna, calibrator, or decision threshold"
        ),
        "profiles": ["legacy", "regularized_v2"],
        "select_features_using_demo": False,
        "external_holdout_independent": False,
        "candidate_release_allowed": False,
        "reference_active_registry_must_remain_unchanged": True,
    }
    payload = json.dumps(metadata, sort_keys=True).encode("utf-8")
    metadata["protocol_sha256"] = hashlib.sha256(payload).hexdigest()
    return manifest, features, quality, bundle, folds, metadata
