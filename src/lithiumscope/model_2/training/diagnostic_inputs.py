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
    older_tables = older_tables or (
        root / "results" / "model_2" / "runs" / OLDER_RUN / "tables"
    )
    previous = {}
    if older_tables.is_dir():
        previous, old_folds, old_y = load_predictions(manifest, older_tables)
        if set(previous) != set(current):
            raise ValueError("Previous and current M2 algorithm sets differ")
        if (not np.array_equal(folds, old_folds) or
            not np.array_equal(y, old_y)):
            raise ValueError("Two competitions lack matching fold labels/order")
    metadata = {
        "historical_rows": min(706, len(manifest)),
        "current_algorithms": sorted(current),
        "previous_competition_available": bool(previous),
        "same_folds_and_labels": bool(previous),
        "comparison_is_reused_development_OOF": True,
        "no_external_holdout": True,
    }
    return manifest, current, previous, {
        "folds": folds, "y": y, "metadata": metadata
    }
