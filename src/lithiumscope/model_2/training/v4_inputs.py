"""Strict spectral-v4 input contract for normal, interactive Model 2 training.

This module only loads and verifies completed real Sentinel-2 data. It NEVER
rebuilds images, trains an estimator, changes labels, or activates a model.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.paths import PROJECT_ROOT
from lithiumscope.model_2.methodology.rededge_training import combine_quality_masks
from lithiumscope.model_2.pipeline import load_model_2_training_frame


ALGORITHMS = (
    "random_forest",
    "extra_trees",
    "hist_gradient_boosting",
    "xgboost",
    "catboost",
    "svm_rbf",
)
RED_EDGE_ASSETS = ("rededge1", "rededge2", "rededge3", "nir08")


def spectral_v4_enabled(cfg: dict) -> bool:
    return bool(cfg.get("training", {}).get("spectral_v4", {}).get("enabled", False))


def spectral_v4_paths(manifest_path: Path, cfg: dict) -> tuple[Path, Path, Path]:
    """Derive one immutable source cohort from the manifest selected by the menu."""
    spec = cfg["training"]["spectral_v4"]
    if int(spec.get("feature_count", 0)) != 170:
        raise ValueError("M2 v4 configuration must declare exactly 170 features")
    if tuple(cfg["model"]["algorithms"]) != ALGORITHMS:
        raise ValueError("M2 v4: all six algorithm families must be enabled")
    root_manifest = PROJECT_ROOT / str(cfg["training"]["manifest_path"])
    expanded_manifest = PROJECT_ROOT / str(cfg["training"]["expansion"]["output_manifest_path"])
    selection = Path(manifest_path).resolve()
    if selection == root_manifest.resolve():
        cohort = "historical_706"
    elif selection == expanded_manifest.resolve():
        if not cfg["training"]["expansion"]["enabled"]:
            raise ValueError("M2 v4: expanded source not enabled")
        cohort = "expanded_783"
    else:
        raise ValueError("M2 v4 refuses an unregistered source manifest")
    entry = spec[cohort]
    directory = PROJECT_ROOT / str(entry)
    if not directory.resolve().is_relative_to(PROJECT_ROOT.resolve()):
        raise ValueError("M2 v4 source must remain under project root")
    return (
        directory / "training_v4_170.csv",
        directory / "build_status.json",
        directory / "rededge_quality.csv",
    )


def verify_spectral_v4_source(manifest_path: Path, cfg: dict) -> tuple[Path, dict]:
    """Check status, SHA, original target/group alignment and scientific QA."""
    matrix, state_file, quality_path = spectral_v4_paths(manifest_path, cfg)
    for file in (matrix, state_file, quality_path):
        if not file.is_file():
            raise FileNotFoundError(
                f"Datos M2 v4 incompletos: {file}. No se entrenará el modelo hasta tener las 170 características."
            )
    state = json.loads(state_file.read_text(encoding="utf-8"))
    manifest = pd.read_csv(manifest_path)
    expected = len(manifest)
    if (
        expected not in (706, 783)
        or state.get("status") != "complete"
        or state.get("cohort") != expected
        or state.get("downloaded") != expected
        or state.get("failed") != 0
        or state.get("feature_extractor_version") != 4
        or state.get("feature_count") != 170
        or state.get("preserved_original_features") != 126
        or state.get("new_real_rededge_features") != 44
    ):
        raise ValueError("M2 v4 source was not fully built and QA-verified")
    if file_sha256(manifest_path) != state.get("manifest_sha256"):
        raise ValueError("M2 v4 manifest drifted since spectral acquisition")
    if file_sha256(matrix) != state.get("matrix_sha256"):
        raise ValueError("M2 v4 matrix hash mismatch; do not use stale data")
    # Return no labels from this source until the full contract is checked.
    return matrix, {
        "source_manifest_sha256": state["manifest_sha256"],
        "source_matrix_sha256": state["matrix_sha256"],
        "quality_report_path": str(quality_path),
        "source_count": expected,
        "feature_extractor_version": 4,
    }


def load_spectral_v4_training_frame(manifest_path: Path, cfg: dict) -> pd.DataFrame:
    matrix, _ = verify_spectral_v4_source(manifest_path, cfg)
    manifest = pd.read_csv(manifest_path)
    data = pd.read_csv(matrix)
    target = str(cfg["training"]["lithium_column"])
    spatial = str(cfg["training"]["spatial_group_column"])
    if (
        data.shape != (len(manifest), 172)
        or list(data.columns[:2]) != [target, spatial]
        or not np.array_equal(data[target].to_numpy(float), manifest[target].to_numpy(float))
        or not np.array_equal(
            data[spatial].astype(str).to_numpy(),
            manifest[spatial].astype(str).to_numpy(),
        )
    ):
        raise ValueError("M2 v4 rows, labels, groups or feature count changed")
    if not data.columns.is_unique or not np.isfinite(data.drop(columns=[target, spatial]).to_numpy(float)).all():
        raise ValueError("M2 v4 features are duplicated or nonfinite")
    # Verify the exact 126 original descriptors, not only their announced count.
    legacy = load_model_2_training_frame(manifest_path)
    if legacy.shape != (len(data), 128):
        raise ValueError("Historical six-band spectral schema is not 126")
    old_features = legacy.drop(columns=[target, spatial])
    new_features = data.drop(columns=[target, spatial])
    # Separate CSV roundtrips produce differences up to ~3e-14. This accepts
    # representation rounding only, never a material scientific change.
    if list(new_features.columns[:126]) != list(old_features.columns) or not np.allclose(
        new_features.iloc[:, :126].to_numpy(float),
        old_features.to_numpy(float),
        rtol=1e-13,
        atol=5e-13,
    ):
        raise ValueError("M2 v4 altered one or more of the 126 base features")
    print(
        "[OK] Modelo 2 v4: 170 características reales verificadas "
        f"({len(data)} muestras; 126 originales + 44 red-edge).",
        flush=True,
    )
    return data


def combine_v4_quality_mask(
    prior_mask: np.ndarray,
    manifest_path: Path,
    cfg: dict,
) -> tuple[np.ndarray, dict]:
    _, _, quality_path = spectral_v4_paths(manifest_path, cfg)
    manifest = pd.read_csv(manifest_path)
    combined, evidence = combine_quality_masks(prior_mask, quality_path, manifest)
    return combined, {
        "enabled": True,
        "manifest_rows": len(manifest),
        "training_eligible": int(combined.sum()),
        "training_excluded": int((~combined).sum()),
        "policy": "all_12_sentinel_bands_train_only_QA95",
        "original_and_new_spectral_quality": evidence,
        "all_outer_test_retained": True,
    }


def build_scientific_training_signature(manifest_path: Path, cfg: dict) -> str:
    """Do not ever reuse a v3 or stale-QA checkpoint for a v4 competition."""
    from lithiumscope.core.reproducibility import canonical_json_hash
    from lithiumscope.core.run_resume import build_training_signature

    if not spectral_v4_enabled(cfg):
        return build_training_signature("model_2", manifest_path, ("app", "model_2"))
    matrix, evidence = verify_spectral_v4_source(manifest_path, cfg)
    base = build_training_signature("model_2", matrix, ("app", "model_2"))
    return canonical_json_hash(
        {
            "matrix_and_code_config_signature": base,
            "rededge_qa_sha256": file_sha256(Path(evidence["quality_report_path"])),
            "original_sixband_qa_sha256": file_sha256(PROJECT_ROOT / str(cfg["quality_training"]["inventory_path"])),
        }
    )
