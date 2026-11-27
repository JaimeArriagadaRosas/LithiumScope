"""Prepare the M2 training matrix with content-addressed, verified caching."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from lithiumscope.core.config import load_config
from lithiumscope.core.logging.session import get_logger
from lithiumscope.core.paths import PROJECT_ROOT
from lithiumscope.core.reproducibility import canonical_json_hash
from lithiumscope.model_2.data.feature_cache import source_identity, read_cache, write_cache
from lithiumscope.model_2.preparation.training_matrix import build_training_set


def resolveresolve_project_path(raw: str, root: Path = PROJECT_ROOT) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else root / path


def load_model_2_training_frame(path: Path | None = None) -> pd.DataFrame:
    logger = get_logger("model_2.preparation.training_data")
    cfg = load_config("model_2")
    training = cfg["training"]
    imagery = cfg["imagery"]
    manifest_path = path or resolve_project_path(str(training["manifest_path"]))
    manifest, identity = source_identity(manifest_path, cfg)
    original = resolve_project_path(str(training["manifest_path"]))
    configured = (training["feature_cache_path"] if manifest_path.resolve() == original.resolve()
                  else training.get("expansion", {}).get("feature_cache_path",
                        "data/processed/model_2/vnext_expanded/spectral_features.csv"))
    base = resolve_project_path(str(configured))
    cache = base.parent / "verified_cache" / f"{canonical_json_hash(identity)}.csv"
    cached = read_cache(cache, manifest, cfg, identity)
    if cached is not None:
        logger.info("Verified feature cache reused: %s (%d rows)", cache, len(cached))
        print(f"[OK] Características espectrales verificadas reutilizadas: {len(cached)} muestras")
        return cached
    print("[INFO] No hay caché íntegra compatible; extrayendo características.")
    logger.info("Extracting M2 training features from %s", manifest_path)
    frame = build_training_set(
        manifest, lithium_column=str(training["lithium_column"]),
        spatial_group_column=str(training.get("spatial_group_column", "spatial_group")),
        band_names=tuple(str(value) for value in imagery["bands"]),
        normalize_per_band=bool(imagery.get("normalize_per_band", False)),
    )
    # Detect source changes during computation before publishing the cache.
    _, final_identity = source_identity(manifest_path, cfg)
    if identity != final_identity:
        logger.error("Source identity changed during feature extraction: %s", manifest_path)
        raise ValueError("Spectral sources changed during feature extraction")
    write_cache(cache, frame, manifest, cfg, identity)
    logger.info("Published verified M2 training features: %s (%d rows)", cache, len(frame))
    return frame
