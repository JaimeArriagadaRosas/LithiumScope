from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1
from uuid import uuid4
from pathlib import Path

import pandas as pd

from lithiumscope.core.config import load_config
from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.logging.session import get_logger
from lithiumscope.core.paths import PROJECT_ROOT
from lithiumscope.model_2.data.sample_source import load_georeferenced_li_samples
from lithiumscope.tools.acquisition.sentinel2.provider import (
    Sentinel2Provider,
    SentinelAssetError,
    SentinelConfig,
    SentinelProviderError,
    SentinelSceneUnavailable,
    download_patch,
)
from lithiumscope.runtime.console_status import Spinner
from lithiumscope.tools.acquisition.sentinel2.configuration import sentinel_config as _sentinel_config

logger = get_logger("model_2.dataset_builder")

_MANIFEST_COLUMNS = [
    "sample_id",
    "Li_icpms",
    "longitude",
    "latitude",
    "spatial_group",
    "image_path",
    "sentinel_scene_id",
    "sentinel_cloud_cover",
]
_FAILURE_COLUMNS = [
    "sample_id",
    "longitude",
    "latitude",
    "category",
    "detail",
]


@dataclass(frozen=True)
class Model2DatasetResult:
    manifest_path: Path
    total_candidates: int
    ready_samples: int
    failed_samples: int
    resumed_samples: int = 0


def _resolve_project_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else PROJECT_ROOT / path



def _cache_name(
    sample_id: str,
    latitude: float,
    longitude: float,
) -> str:
    identity = (
        f"{sample_id}|{latitude:.12f}|{longitude:.12f}"
    ).encode("utf-8")
    suffix = sha1(identity, usedforsecurity=False).hexdigest()[:10]
    return f"{sample_id}_{suffix}.npy"


def _sample_key(
    sample_id: str,
    latitude: float,
    longitude: float,
) -> str:
    return f"{sample_id}|{latitude:.12f}|{longitude:.12f}"


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        frame.to_csv(temporary, index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_valid_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=_MANIFEST_COLUMNS)
    try:
        frame = pd.read_csv(path, dtype={"sample_id":"string","sentinel_scene_id":"string"})
    except Exception:
        logger.warning(
            "Could not read Model 2 checkpoint %s; ignoring it",
            path,
            exc_info=True,
        )
        return pd.DataFrame(columns=_MANIFEST_COLUMNS)

    if "image_path" not in frame.columns:
        return pd.DataFrame(columns=_MANIFEST_COLUMNS)

    # Byte/geometry/identity verification follows in reconcile_checkpoint.
    valid = frame.copy()
    for column in _MANIFEST_COLUMNS:
        if column not in valid.columns:
            valid[column] = pd.NA
    return valid[_MANIFEST_COLUMNS]


def _persist_progress(
    rows: dict[str, dict],
    failures: list[dict],
    partial_manifest_path: Path,
    failures_path: Path,
) -> None:
    manifest = pd.DataFrame(
        list(rows.values()),
        columns=_MANIFEST_COLUMNS,
    )
    failure_frame = pd.DataFrame(
        failures,
        columns=_FAILURE_COLUMNS,
    )
    _atomic_write_csv(manifest, partial_manifest_path)
    _atomic_write_csv(failure_frame, failures_path)


def _progress_message(
    done: int,
    total: int,
    ready: int,
    failed: int,
    resumed: int,
) -> str:
    return (
        f"Modelo 2 · Sentinel-2 {done}/{total} | "
        f"listos={ready} | omitidos={failed} | cache={resumed}"
    )



from lithiumscope.model_2.data.builder_assemble import ensure_model_2_dataset
