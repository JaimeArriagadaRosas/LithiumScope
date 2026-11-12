"""Isolated responsibility with unchanged original API."""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lithiumscope.model_2.data.sentinel2 import Path, Sentinel2Provider, np


def _read_patch(
    href: str,
    latitude: float,
    longitude: float,
    patch_size_m: float,
    patch_pixels: int,
    *,
    categorical: bool = False,
) -> np.ndarray:
    from lithiumscope.model_2.data.sentinel2 import SentinelAssetError, SentinelProviderError, _masked_to_float, np
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
                    resampling=(
                        Resampling.nearest if categorical else Resampling.bilinear
                    ),
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
    *,
    selected_scene: dict | None = None,
    scientific_quality: dict | None = None,
) -> tuple[Path, dict]:
    from lithiumscope.model_2.data.sentinel2 import Path, Sentinel2Provider, SentinelAssetError, _asset_href, _read_patch, json, np
    if destination.exists() and destination.stat().st_size > 0:
        metadata_path = destination.with_suffix(".json")
        metadata = {}
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return destination, metadata

    scene = (selected_scene if selected_scene is not None
             else provider.search_best_scene(latitude, longitude))
    if scientific_quality is not None:
        if not scientific_quality.get("scientifically_accepted_proxy"):
            raise SentinelAssetError("Scientific scene selection has not passed QA")
        if str(scientific_quality.get("scene_id")) != str(scene.get("id")):
            raise SentinelAssetError("Scientific QA belongs to a different scene")
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
    if scientific_quality is not None:
        # Metadata is provenance of technical QA, not field verification.
        metadata["scientific_patch_qa"] = scientific_quality
    destination.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return destination, metadata
