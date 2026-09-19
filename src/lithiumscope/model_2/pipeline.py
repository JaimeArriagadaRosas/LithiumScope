from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from lithiumscope.core.config import load_config
from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.paths import PROJECT_ROOT
from lithiumscope.core.reproducibility import canonical_json_hash
from lithiumscope.model_2.steps.step_05_spectral_features import (
    FEATURE_EXTRACTOR_VERSION,
)
from lithiumscope.model_2.steps.step_07_build_training_set import (
    build_training_set,
)


def _project_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _cache_signature(
    manifest_path: Path,
    config: dict,
) -> str:
    imagery = config["imagery"]
    payload = {
        "manifest_sha256": file_sha256(manifest_path),
        "feature_extractor_version": FEATURE_EXTRACTOR_VERSION,
        "bands": list(imagery["bands"]),
        "normalize_per_band": bool(
            imagery.get("normalize_per_band", False)
        ),
    }
    return canonical_json_hash(payload)


def _read_cached_features(
    cache_path: Path,
    metadata_path: Path,
    signature: str,
) -> pd.DataFrame | None:
    if not cache_path.exists() or not metadata_path.exists():
        return None

    try:
        metadata = json.loads(
            metadata_path.read_text(encoding="utf-8")
        )
        if metadata.get("signature") != signature:
            return None
        frame = pd.read_csv(cache_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None

    return frame if not frame.empty else None


def _write_cache(
    frame: pd.DataFrame,
    cache_path: Path,
    metadata_path: Path,
    signature: str,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(cache_path)

    metadata_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "signature": signature,
                "rows": len(frame),
                "columns": list(frame.columns),
                "feature_extractor_version": FEATURE_EXTRACTOR_VERSION,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def load_model_2_training_frame(
    path: Path | None = None,
) -> pd.DataFrame:
    cfg = load_config("model_2")
    training_cfg = cfg["training"]
    imagery_cfg = cfg["imagery"]
    manifest_path = path or _project_path(
        str(training_cfg["manifest_path"])
    )
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Model 2 manifest not found: {manifest_path}. "
            "Debe contener una fila por muestra conocida y una columna image_path."
        )

    cache_path = _project_path(
        str(training_cfg["feature_cache_path"])
    )
    metadata_path = cache_path.with_suffix(
        cache_path.suffix + ".meta.json"
    )
    signature = _cache_signature(
        manifest_path,
        cfg,
    )
    cached = _read_cached_features(
        cache_path,
        metadata_path,
        signature,
    )
    if cached is not None:
        print(
            "[OK] Características espectrales reutilizadas: "
            f"{len(cached)} muestras"
        )
        return cached

    manifest = pd.read_csv(manifest_path)
    frame = build_training_set(
        manifest,
        lithium_column=str(training_cfg["lithium_column"]),
        spatial_group_column=str(
            training_cfg.get(
                "spatial_group_column",
                "spatial_group",
            )
        ),
        band_names=tuple(
            str(value)
            for value in imagery_cfg["bands"]
        ),
        normalize_per_band=bool(
            imagery_cfg.get(
                "normalize_per_band",
                False,
            )
        ),
    )
    _write_cache(
        frame,
        cache_path,
        metadata_path,
        signature,
    )
    return frame
