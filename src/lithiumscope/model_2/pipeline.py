"""Production M2 feature preparation, with verified and content-addressed cache."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from lithiumscope.core.config import load_config
from lithiumscope.core.paths import PROJECT_ROOT
from lithiumscope.core.reproducibility import canonical_json_hash
from lithiumscope.model_2.data.feature_cache import source_identity, read_cache, write_cache
from lithiumscope.model_2.preparation.training_matrix import build_training_set


def _project_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_model_2_training_frame(path: Path | None = None) -> pd.DataFrame:
    cfg = load_config("model_2")
    training = cfg["training"]
    imagery = cfg["imagery"]
    manifest_path = path or _project_path(str(training["manifest_path"]))
    manifest, identity = source_identity(manifest_path, cfg)
    original = _project_path(str(training["manifest_path"]))
    configured = (training["feature_cache_path"] if manifest_path.resolve() == original.resolve()
                  else training.get("expansion", {}).get("feature_cache_path",
                        "data/processed/model_2/vnext_expanded/spectral_features.csv"))
    base = _project_path(str(configured))
    cache = base.parent / "verified_cache" / f"{canonical_json_hash(identity)}.csv"
    cached = read_cache(cache, manifest, cfg, identity)
    if cached is not None:
        print(f"[OK] Características espectrales verificadas reutilizadas: {len(cached)} muestras")
        return cached
    print("[INFO] No hay caché íntegra compatible; extrayendo características.")
    frame = build_training_set(
        manifest, lithium_column=str(training["lithium_column"]),
        spatial_group_column=str(training.get("spatial_group_column", "spatial_group")),
        band_names=tuple(str(value) for value in imagery["bands"]),
        normalize_per_band=bool(imagery.get("normalize_per_band", False)),
    )
    # Detect source changes during computation before publishing the cache.
    _, final_identity = source_identity(manifest_path, cfg)
    if identity != final_identity:
        raise ValueError("Spectral sources changed during feature extraction")
    write_cache(cache, frame, manifest, cfg, identity)
    return frame


def run_training(device, *, manifest=None):
    """M2 owns source resolution independently of M1 training."""
    from lithiumscope.runtime.preboot import require_full_training_environment
    from lithiumscope.runtime.dataset_provisioning import require_model_2_expanded_manifest
    from lithiumscope.model_2.training.competition import run_model_2_competition
    require_full_training_environment("model_2")
    source = manifest if manifest is not None else require_model_2_expanded_manifest()
    return run_model_2_competition(source, device)


def run_independent_evaluation(phase, root, **kwargs):
    """Separate registered holdout phases; no automatic activation."""
    from lithiumscope.evaluation.holdout.workflow import run_holdout_phase
    return run_holdout_phase("model_2",phase,root,**kwargs)
