from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math

import numpy as np
import requests

from lithiumscope.core.logger import get_logger

logger = get_logger("model_2.sentinel2")


@dataclass(frozen=True)
class SentinelConfig:
    stac_url: str
    collection: str
    datetime: str
    cloud_cover_max: float
    search_limit: int
    patch_size_m: float
    patch_pixels: int
    bands: tuple[str, ...]


def _cloud_cover(item: dict) -> float:
    value = item.get("properties", {}).get("eo:cloud_cover")
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.inf


def search_best_scene(
    latitude: float,
    longitude: float,
    config: SentinelConfig,
    session: requests.Session | None = None,
) -> dict:
    session = session or requests.Session()
    delta = 0.001
    payload = {
        "collections": [config.collection],
        "bbox": [
            longitude - delta,
            latitude - delta,
            longitude + delta,
            latitude + delta,
        ],
        "datetime": config.datetime,
        "limit": config.search_limit,
        "query": {"eo:cloud_cover": {"lte": config.cloud_cover_max}},
    }
    response = session.post(config.stac_url, json=payload, timeout=60)
    response.raise_for_status()
    features = response.json().get("features", [])
    if not features:
        raise RuntimeError(
            f"Sentinel-2: no se encontraron escenas para ({latitude:.5f}, {longitude:.5f})."
        )
    return min(features, key=_cloud_cover)


def _asset_href(scene: dict, band: str) -> str:
    asset = scene.get("assets", {}).get(band)
    if not asset or not asset.get("href"):
        raise RuntimeError(f"Sentinel-2 scene {scene.get('id')} no contiene el asset '{band}'.")
    return str(asset["href"])


def _read_patch(
    href: str,
    latitude: float,
    longitude: float,
    patch_size_m: float,
    patch_pixels: int,
) -> np.ndarray:
    try:
        import rasterio
        from rasterio.enums import Resampling
        from rasterio.windows import from_bounds
        from rasterio.warp import transform
    except ImportError as exc:
        raise RuntimeError(
            'Rasterio es necesario para preparar Sentinel-2. Instale: pip install -e ".[imagery]"'
        ) from exc

    with rasterio.Env(
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.TIF,.tiff,.TIFF",
    ):
        with rasterio.open(href) as dataset:
            if dataset.crs is None:
                raise RuntimeError(f"Asset Sentinel-2 sin CRS: {href}")
            xs, ys = transform("EPSG:4326", dataset.crs, [longitude], [latitude])
            x = float(xs[0])
            y = float(ys[0])
            half = patch_size_m / 2.0
            window = from_bounds(x - half, y - half, x + half, y + half, dataset.transform)
            data = dataset.read(
                1,
                window=window,
                out_shape=(patch_pixels, patch_pixels),
                boundless=True,
                masked=True,
                resampling=Resampling.bilinear,
            )
            return np.asarray(data.filled(np.nan), dtype=np.float32)


def download_patch(
    sample_id: str,
    latitude: float,
    longitude: float,
    destination: Path,
    config: SentinelConfig,
    session: requests.Session | None = None,
) -> tuple[Path, dict]:
    if destination.exists() and destination.stat().st_size > 0:
        metadata_path = destination.with_suffix(".json")
        metadata = {}
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return destination, metadata

    scene = search_best_scene(latitude, longitude, config, session=session)
    arrays = [
        _read_patch(
            _asset_href(scene, band),
            latitude,
            longitude,
            config.patch_size_m,
            config.patch_pixels,
        )
        for band in config.bands
    ]
    stack = np.stack(arrays, axis=0)
    if not np.isfinite(stack).any():
        raise RuntimeError(f"Sentinel-2 patch vacío para muestra {sample_id}.")

    destination.parent.mkdir(parents=True, exist_ok=True)
    np.save(destination, stack)

    metadata = {
        "sample_id": sample_id,
        "latitude": latitude,
        "longitude": longitude,
        "scene_id": scene.get("id"),
        "datetime": scene.get("properties", {}).get("datetime"),
        "cloud_cover": scene.get("properties", {}).get("eo:cloud_cover"),
        "bands": list(config.bands),
        "collection": scene.get("collection"),
    }
    destination.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return destination, metadata
