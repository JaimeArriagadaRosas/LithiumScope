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

    def search_candidate_scenes(
        self, latitude: float, longitude: float, *,
        datetime: str | None = None,
        max_items: int | None = None,
    ) -> list[dict]:
        """Search all scenes intersecting the whole patch, for local QC.

        Unlike legacy best-scene selection this routine never assumes that
        low global EO cloud implies actual patch coverage.
        A narrower time window can be frozen for same-date selection.
        """
        if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
            raise ValueError("Invalid geographical point")
        limit = max_items if max_items is not None else self.config.search_limit
        if limit < 1:
            raise ValueError("Search limit must be positive")
        # Bounding box should cover the patch's whole half-width at this latitude.
        half_lat = self.config.patch_size_m / (2.0 * 110_540)
        half_lon = self.config.patch_size_m / (
            2.0 * 111_320 * max(0.01, math.cos(math.radians(latitude)))
        )
        client = self._open_client()
        try:
            search = client.search(
                collections=[self.config.collection],
                bbox=[longitude-half_lon, latitude-half_lat,
                      longitude+half_lon, latitude+half_lat],
                datetime=datetime or self.config.datetime,
                query={"eo:cloud_cover": {"lte": self.config.cloud_cover_max}},
                max_items=limit,
            )
            rows = [_item_as_dict(v) for v in search.items()]
        except Exception as exc:
            raise _provider_error("búsqueda local de candidatos", exc) from exc
        rows.sort(key=lambda x: (_cloud_cover(x), str(x.get("id",""))))
        return rows

    def search_validated_scene(
        self, latitude: float, longitude: float, *,
        datetime: str | None = None,
        max_candidates: int | None = None,
        policy=None,
    ) -> tuple[dict, dict, list[dict]]:
        """Opt-in full six-band and SCL local selection. No changes to caches."""
        from lithiumscope.model_2.methodology.satellite_quality import (
            ScientificScenePolicy, choose_scene, SPECTRAL_BANDS
        )
        if tuple(self.config.bands) != SPECTRAL_BANDS:
            raise ValueError("Scientific selector requires frozen six-band schema")
        if not datetime or "/" not in datetime:
            raise ValueError(
                "Scientific selection requires an explicit pre-registered "
                "datetime interval; it cannot search all years implicitly")
        policy = policy or ScientificScenePolicy(
            max_scene_cloud=self.config.cloud_cover_max,
            max_candidates=min(self.config.search_limit,20),
        )
        items = self.search_candidate_scenes(
            latitude,longitude,datetime=datetime,
            max_items=max_candidates or policy.max_candidates,
        )
        if not items:
            raise SentinelSceneUnavailable("No Sentinel-2 candidates in requested window")
        selected, quality, audits=choose_scene(
            items,latitude,longitude,policy=policy,
            patch_size_m=self.config.patch_size_m,
            patch_pixels=self.config.patch_pixels,
        )
        if selected is None:
            raise SentinelSceneUnavailable(
                "No candidate passes six-band coverage and local SCL; "
                + "; ".join(
                    str(x["scene_id"])+":"+
                    ",".join(x.get("rejection_reasons",[]))
                    for x in audits[:8])
            )
        return selected, quality, audits

    def search_best_scene(self, latitude: float, longitude: float) -> dict:
        """Legacy rule kept byte-for-byte compatible with frozen model datasets."""
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





from lithiumscope.model_2.data.sentinel2_patch_io import _read_patch, download_patch
