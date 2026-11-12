from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path

import pandas as pd

from lithiumscope.core.config import load_config
from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.logger import get_logger
from lithiumscope.core.paths import PROJECT_ROOT
from lithiumscope.model_2.data.sample_source import load_georeferenced_li_samples
from lithiumscope.model_2.data.sentinel2 import (
    Sentinel2Provider,
    SentinelAssetError,
    SentinelConfig,
    SentinelProviderError,
    SentinelSceneUnavailable,
    download_patch,
)
from lithiumscope.runtime.console_status import Spinner

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


def _sentinel_config(config: dict) -> SentinelConfig:
    imagery = config["imagery"]
    return SentinelConfig(
        stac_url=str(imagery["stac_url"]),
        collection=str(imagery["collection"]),
        datetime=str(imagery["datetime"]),
        cloud_cover_max=float(imagery["cloud_cover_max"]),
        search_limit=int(imagery["search_limit"]),
        request_timeout_seconds=float(
            imagery.get("request_timeout_seconds", 60)
        ),
        max_retries=int(imagery.get("max_retries", 3)),
        scene_cache_decimals=int(
            imagery.get("scene_cache_decimals", 4)
        ),
        patch_size_m=float(imagery["patch_size_m"]),
        patch_pixels=int(imagery["patch_pixels"]),
        bands=tuple(str(value) for value in imagery["bands"]),
    )


def _cache_name(
    sample_id: str,
    latitude: float,
    longitude: float,
) -> str:
    identity = (
        f"{sample_id}|{latitude:.6f}|{longitude:.6f}"
    ).encode("utf-8")
    suffix = sha1(identity, usedforsecurity=False).hexdigest()[:10]
    return f"{sample_id}_{suffix}.npy"


def _sample_key(
    sample_id: str,
    latitude: float,
    longitude: float,
) -> str:
    return f"{sample_id}|{latitude:.6f}|{longitude:.6f}"


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _load_valid_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=_MANIFEST_COLUMNS)
    try:
        frame = pd.read_csv(path)
    except Exception:
        logger.warning(
            "Could not read Model 2 checkpoint %s; ignoring it",
            path,
            exc_info=True,
        )
        return pd.DataFrame(columns=_MANIFEST_COLUMNS)

    if "image_path" not in frame.columns:
        return pd.DataFrame(columns=_MANIFEST_COLUMNS)

    valid = frame[
        frame["image_path"].map(
            lambda value: Path(str(value)).exists()
        )
    ].copy()
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
