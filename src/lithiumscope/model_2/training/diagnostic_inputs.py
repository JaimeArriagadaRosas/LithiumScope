"""Read-only audit of frozen M2 competition predictions and release artifact."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from lithiumscope.core.paths import PROJECT_ROOT
from lithiumscope.model_2.training.postprocess import load_predictions

RELEASE_RUN = "training_20260920_195933_m0300"
OLDER_RUN = "training_20261111_170832_m0300"
OLDER_REFERENCE = (
    "results/model_2/runs/training_20261112_112450_m0300/"
    "all_reference_paired_oof.csv"
)
SIX_ALGORITHMS = frozenset({
    "catboost", "extra_trees", "hist_gradient_boosting",
    "random_forest", "svm_rbf", "xgboost",
})


def release_contract(root: Path) -> dict:
    """Actually load the trusted CatBoost joblib; never use training-set scores."""
    folder = root / "models" / "model_2" / "trained" / RELEASE_RUN
    meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    schema = json.loads((folder / "feature_schema.json").read_text(encoding="utf-8"))
    model_file = folder / "model.joblib"
    digest = hashlib.sha256(model_file.read_bytes()).hexdigest()
    if digest != meta["model_sha256"]:
        raise ValueError("Published M2 release hash mismatch")
    bundle = joblib.load(model_file)
    if (meta.get("algorithm") != "catboost" or
        len(schema["features"]) != 126 or
        bundle.get("features") != schema["features"] or
        bundle.get("algorithm") != "catboost" or
        not hasattr(bundle.get("estimator"), "predict_proba")):
        raise ValueError("Published M2 release bundle/schema mismatch")
    registry = json.loads((root / "models/active_models.json").read_text(encoding="utf-8"))
    active = registry["models"]["model_2"]["model_path"].replace("\\", "/")
    if RELEASE_RUN not in active:
        raise ValueError("Published M2 reference no longer active")
    return {
        "run": RELEASE_RUN, "algorithm": "catboost",
        "model_sha256": digest, "features": len(schema["features"]),
        "loaded_estimator": type(bundle["estimator"]).__name__,
        "operating_threshold": float(meta["operating_threshold"]),
        "reference_policy": (
            "Never score fitted release on its own training examples; "
            "reference comparison uses original protocol retrained OOF."
        ),
    }


def loaded_competitions(
    manifest_path: Path,
    current_tables: Path,
    root: Path = PROJECT_ROOT,
    older_tables: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray], dict]:
    manifest = pd.read_csv(manifest_path)
    if manifest.sample_id.duplicated().any() or manifest.sample_id.isna().any():
        raise ValueError("Sample IDs must be unique and present")
    current, folds, y = load_predictions(manifest, current_tables)
    if len(current) < 2:
        raise ValueError("At least two completed algorithms are needed")
    if set(current) - SIX_ALGORITHMS:
        raise ValueError("Unknown M2 competition algorithms")
    compare_legacy = len(manifest) == 706
    older_run = (
        "training_20261113_114607_m0300"
        if compare_legacy else OLDER_RUN
    )
    older_tables = older_tables or (
        root / "results" / "model_2" / "runs" / older_run / "tables"
    )
    previous = {}
    prior_n = None
    if older_tables.is_dir():
        if compare_legacy:
            # This is the desired 706-vs-783 ablation: compare only common
            # historical held-out cases, whose fold IDs/labels are identical.
            prior_manifest = pd.read_csv(
                root / "data/processed/model_2/vnext_expanded/training_manifest.csv"
            )
            if len(prior_manifest) != 783 or not np.array_equal(
                manifest.sample_id.astype(str),
                prior_manifest.sample_id.iloc[:706].astype(str)
            ) or not np.allclose(
                manifest.Li_icpms.to_numpy(float),
                prior_manifest.Li_icpms.iloc[:706].to_numpy(float)
            ) or not np.array_equal(
                manifest.spatial_group.astype(str),
                prior_manifest.spatial_group.iloc[:706].astype(str)
            ):
                raise ValueError("Prior 783 records do not preserve frozen 706")
        else:
            prior_manifest = manifest
        old_scores, old_folds, old_y = load_predictions(
            prior_manifest, older_tables
        )
        prior_n = len(prior_manifest)
        if compare_legacy:
            old_scores = {key: values[:706] for key, values in old_scores.items()}
            old_folds = old_folds[:706]
            old_y = old_y[:706]
        if set(old_scores) != set(current):
            raise ValueError("Previous and current M2 algorithm sets differ")
        if (not np.array_equal(folds, old_folds) or
            not np.array_equal(y, old_y)):
            raise ValueError("Two competitions lack matching fold labels/order")
        previous = old_scores
    metadata = {
        "historical_rows": min(706, len(manifest)),
        "current_training_source_rows": len(manifest),
        "previous_training_source_rows": prior_n,
        "previous_run": older_run if previous else None,
        "historical_706_vs_expanded_783_ablation": (
            compare_legacy and prior_n == 783
        ),
        "current_algorithms": sorted(current),
        "previous_competition_available": bool(previous),
        "same_folds_and_labels": bool(previous),
        "comparison_is_reused_development_OOF": True,
        "no_external_holdout": True,
    }
    return manifest, current, previous, {
        "folds": folds, "y": y, "metadata": metadata
    }
