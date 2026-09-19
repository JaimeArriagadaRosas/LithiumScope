from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math

import numpy as np

from lithiumscope.core.logger import get_logger

logger = get_logger("model_2.sentinel2")


class SentinelProviderError(RuntimeError):
    """Provider-wide error. Repeating the same query for every sample is pointless."""


class SentinelSceneUnavailable(RuntimeError):
    """No suitable Sentinel-2 scene exists for one sample."""


class SentinelAssetError(RuntimeError):
    """A scene exists, but its raster assets cannot be read for one sample."""


@dataclass(frozen=True)
class SentinelConfig:
    stac_url: str
    collection: str
    datetime: str
    cloud_cover_max: float
    search_limit: int
    request_timeout_seconds: float
    max_retries: int
    scene_cache_decimals: int
    patch_size_m: float
    patch_pixels: int
    bands: tuple[str, ...]


def _cloud_cover(item: dict) -> float:
    value = item.get("properties", {}).get("eo:cloud_cover")
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.inf


def _item_as_dict(item) -> dict:
    if isinstance(item, dict):
        return item
    if hasattr(item, "to_dict"):
        return item.to_dict()
    raise TypeError(f"Unsupported STAC item type: {type(item)!r}")


def _error_status(exc: BaseException) -> int | None:
    value = getattr(exc, "status_code", None)
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _provider_error(operation: str, exc: BaseException) -> SentinelProviderError:
    status = _error_status(exc)
    status_text = f" HTTP {status}" if status is not None else ""
    detail = str(exc).strip() or exc.__class__.__name__
    return SentinelProviderError(
        f"Earth Search falló durante {operation}.{status_text} Detalle: {detail}"
    )


def _masked_to_float(data) -> np.ndarray:
    """Convert masked integer raster data before filling nodata with NaN."""
    masked = np.ma.asarray(data).astype(np.float32)
    return np.asarray(masked.filled(np.nan), dtype=np.float32)


class Sentinel2Provider:
    """Small adapter around pystac-client with query caching and fail-fast errors."""

    def __init__(self, config: SentinelConfig, client=None) -> None:
        self.config = config
        self._client = client
        self._scene_cache: dict[tuple[float, float], dict] = {}

    def _open_client(self):
        if self._client is not None:
            return self._client
        try:
            from pystac_client import Client
            from pystac_client.stac_api_io import StacApiIO
        except ImportError as exc:
            raise SentinelProviderError(
                'pystac-client no está instalado. Ejecute: pip install -e ".[imagery]"'
            ) from exc

        try:
            stac_io = StacApiIO(
                timeout=self.config.request_timeout_seconds,
                max_retries=self.config.max_retries,
            )
            self._client = Client.open(self.config.stac_url, stac_io=stac_io)
            return self._client
        except Exception as exc:
            raise _provider_error("apertura del catálogo STAC", exc) from exc

    def validate(self) -> None:
        client = self._open_client()
        try:
            client.get_collection(self.config.collection)
        except Exception as exc:
            raise _provider_error(
                f"validación de la colección {self.config.collection}",
                exc,
            ) from exc
        logger.info(
            "Earth Search provider validated endpoint=%s collection=%s",
            self.config.stac_url,
            self.config.collection,
        )

    def search_best_scene(self, latitude: float, longitude: float) -> dict:
        decimals = self.config.scene_cache_decimals
        cache_key = (round(latitude, decimals), round(longitude, decimals))
        cached = self._scene_cache.get(cache_key)
        if cached is not None:
            return cached

        delta = 0.001
        client = self._open_client()
        try:
            search = client.search(
                collections=[self.config.collection],
                bbox=[
                    longitude - delta,
                    latitude - delta,
                    longitude + delta,
                    latitude + delta,
                ],
                datetime=self.config.datetime,
                query={"eo:cloud_cover": {"lte": self.config.cloud_cover_max}},
                max_items=self.config.search_limit,
            )
            items = [_item_as_dict(item) for item in search.items()]
        except Exception as exc:
            raise _provider_error("búsqueda de escenas", exc) from exc

        if not items:
            raise SentinelSceneUnavailable(
                f"No se encontraron escenas Sentinel-2 para ({latitude:.5f}, {longitude:.5f}) "
                f"con nubosidad <= {self.config.cloud_cover_max:.0f}%."
            )

        best = min(items, key=_cloud_cover)
        self._scene_cache[cache_key] = best
        return best


def _asset_href(scene: dict, band: str) -> str:
    asset = scene.get("assets", {}).get(band)
    if not asset or not asset.get("href"):
        raise SentinelAssetError(
            f"Sentinel-2 scene {scene.get('id')} no contiene el asset '{band}'."
        )
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
        raise SentinelProviderError(
            'Rasterio es necesario para preparar Sentinel-2. Instale: pip install -e ".[imagery]"'
        ) from exc

    try:
        with rasterio.Env(
            AWS_NO_SIGN_REQUEST="YES",
            GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
            CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.TIF,.tiff,.TIFF",
            GDAL_HTTP_MAX_RETRY="3",
            GDAL_HTTP_RETRY_DELAY="1",
            GDAL_HTTP_CONNECTTIMEOUT="20",
            GDAL_HTTP_TIMEOUT="60",
        ):
            with rasterio.open(href) as dataset:
                if dataset.crs is None:
                    raise SentinelAssetError(f"Asset Sentinel-2 sin CRS: {href}")
                xs, ys = transform("EPSG:4326", dataset.crs, [longitude], [latitude])
                x = float(xs[0])
                y = float(ys[0])
                half = patch_size_m / 2.0
                window = from_bounds(
                    x - half,
                    y - half,
                    x + half,
                    y + half,
                    dataset.transform,
                )
                data = dataset.read(
                    1,
                    window=window,
                    out_shape=(patch_pixels, patch_pixels),
                    boundless=True,
                    masked=True,
                    resampling=Resampling.bilinear,
                )
                return _masked_to_float(data)
    except SentinelAssetError:
        raise
    except Exception as exc:
        raise SentinelAssetError(f"No se pudo leer asset Sentinel-2 {href}: {exc}") from exc


def download_patch(
    sample_id: str,
    latitude: float,
    longitude: float,
    destination: Path,
    provider: Sentinel2Provider,
) -> tuple[Path, dict]:
    if destination.exists() and destination.stat().st_size > 0:
        metadata_path = destination.with_suffix(".json")
        metadata = {}
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return destination, metadata

    scene = provider.search_best_scene(latitude, longitude)
    config = provider.config
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
        raise SentinelAssetError(f"Sentinel-2 patch vacío para muestra {sample_id}.")

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
