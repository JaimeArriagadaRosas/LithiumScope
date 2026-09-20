from __future__ import annotations

from pathlib import Path

import pandas as pd

from lithiumscope.core.config import load_config
from lithiumscope.core.logger import get_logger
from lithiumscope.model_2.data.sentinel2 import (
    Sentinel2Provider,
    SentinelAssetError,
    SentinelConfig,
    SentinelProviderError,
    SentinelSceneUnavailable,
    download_patch,
)
from lithiumscope.runtime.console_status import Spinner

logger = get_logger("prediction.imagery")


def _sentinel_config(datetime_override: str | None = None) -> SentinelConfig:
    cfg = load_config("model_2")["imagery"]
    return SentinelConfig(
        stac_url=str(cfg["stac_url"]),
        collection=str(cfg["collection"]),
        datetime=str(datetime_override or cfg["datetime"]),
        cloud_cover_max=float(cfg["cloud_cover_max"]),
        search_limit=int(cfg["search_limit"]),
        request_timeout_seconds=float(cfg.get("request_timeout_seconds", 60)),
        max_retries=int(cfg.get("max_retries", 3)),
        scene_cache_decimals=int(cfg.get("scene_cache_decimals", 4)),
        patch_size_m=float(cfg["patch_size_m"]),
        patch_pixels=int(cfg["patch_pixels"]),
        bands=tuple(str(value) for value in cfg["bands"]),
    )


def _resolve_coordinate_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    normalized = {str(column).strip().lower(): column for column in frame.columns}
    for candidate in candidates:
        match = normalized.get(candidate.lower())
        if match is not None:
            return str(match)
    return None


def prepare_case_images(
    cases: pd.DataFrame,
    cache_dir: Path,
    *,
    datetime_override: str | None = None,
) -> pd.DataFrame:
    longitude_column = _resolve_coordinate_column(
        cases,
        ("Longitude", "Longitude (X)", "longitude", "lon", "lng"),
    )
    latitude_column = _resolve_coordinate_column(
        cases,
        ("Latitude", "Latitude (Y)", "latitude", "lat"),
    )
    if longitude_column is None or latitude_column is None:
        raise ValueError(
            "Para descargar Sentinel-2 se requieren columnas de longitud y latitud."
        )

    cache_dir.mkdir(parents=True, exist_ok=True)
    provider = Sentinel2Provider(_sentinel_config(datetime_override))
    provider.validate()

    rows: list[dict] = []
    total = len(cases)
    spinner = Spinner(f"Sentinel-2 · preparando 0/{total} casos").start()
    try:
        for position, (_, row) in enumerate(cases.iterrows(), start=1):
            case_id = str(row["case_id"])
            longitude = float(row[longitude_column])
            latitude = float(row[latitude_column])
            destination = cache_dir / f"{case_id}.npy"
            record = {
                "case_id": case_id,
                "longitude": longitude,
                "latitude": latitude,
                "Li_icpms": row.get("Li_icpms"),
                "image_path": None,
                "sentinel_status": "failed",
                "sentinel_scene_id": None,
                "sentinel_cloud_cover": None,
                "sentinel_datetime": None,
                "sentinel_error": None,
            }
            try:
                patch, metadata = download_patch(
                    sample_id=case_id,
                    latitude=latitude,
                    longitude=longitude,
                    destination=destination,
                    provider=provider,
                )
                record.update(
                    {
                        "image_path": str(patch),
                        "sentinel_status": "ready",
                        "sentinel_scene_id": metadata.get("scene_id"),
                        "sentinel_cloud_cover": metadata.get("cloud_cover"),
                        "sentinel_datetime": metadata.get("datetime"),
                    }
                )
            except SentinelSceneUnavailable as exc:
                record["sentinel_status"] = "no_scene"
                record["sentinel_error"] = str(exc)
                logger.warning("No Sentinel scene for case=%s: %s", case_id, exc)
            except SentinelAssetError as exc:
                record["sentinel_status"] = "asset_error"
                record["sentinel_error"] = str(exc)
                logger.warning("Sentinel asset failed for case=%s: %s", case_id, exc)
            rows.append(record)
            spinner.update(
                f"Sentinel-2 · preparando {position}/{total} casos | "
                f"listos={sum(item['sentinel_status'] == 'ready' for item in rows)}"
            )
    except SentinelProviderError:
        spinner.fail("Sentinel-2 · proveedor no disponible")
        raise
    except BaseException:
        spinner.fail("Sentinel-2 · preparación interrumpida")
        raise
    else:
        ready = sum(item["sentinel_status"] == "ready" for item in rows)
        spinner.succeed(f"Sentinel-2 · {ready}/{total} casos listos")
    return pd.DataFrame(rows)
