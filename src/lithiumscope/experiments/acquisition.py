"""Reproducible Sentinel-2 acquisition for evaluation only (never training)."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lithiumscope.core.config import load_config
from lithiumscope.model_2.data.builder import _sentinel_config
from lithiumscope.model_2.data.sentinel2 import Sentinel2Provider, download_patch, _read_patch


# Coordinates are pilot regions, not labelled positive/negative Li samples.
PILOT_LOCATIONS = {
    "atacama": (-23.10, -68.18),
    "tara": (-23.08, -67.33),
    "maricunga": (-26.91, -69.08),
    "pedernales": (-26.25, -68.60),
}
YEARS = (2024, 2025)


def read_scene_quality(scene: dict, latitude: float, longitude: float,
                       size_m: float, pixels: int) -> dict:
    """Use categorical nearest-neighbour SCL; do not interpolate class labels."""
    try:
        import rasterio
        from rasterio.enums import Resampling
        from rasterio.warp import transform
        from rasterio.windows import from_bounds
    except ImportError as exc:
        raise RuntimeError("SCL quality requires rasterio.") from exc

    asset = scene.get("assets", {}).get("scl", {})
    href = asset.get("href")
    if not href:
        return {"quality_status": "NO_SCL", "local_cloud_fraction": None}
    with rasterio.Env(AWS_NO_SIGN_REQUEST="YES", GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                      GDAL_HTTP_TIMEOUT="45", GDAL_HTTP_MAX_RETRY="2"):
        with rasterio.open(href) as source:
            x, y = transform("EPSG:4326", source.crs, [longitude], [latitude])
            half = size_m / 2
            window = from_bounds(x[0]-half, y[0]-half, x[0]+half, y[0]+half, source.transform)
            scl = source.read(1, window=window, out_shape=(pixels, pixels),
                              boundless=True, masked=True, resampling=Resampling.nearest)
            invalid = np.ma.getmaskarray(scl) | np.isin(np.ma.getdata(scl), [0, 1, 3, 7, 8, 9, 10, 11])
            fraction = float(np.mean(invalid))
    return {"quality_status": "ACCEPT" if fraction <= .35 else "REJECT_CLOUD",
            "local_cloud_fraction": fraction}


def acquire_pilot(destination: Path, years=YEARS, locations=PILOT_LOCATIONS,
                  max_patches: int = 40) -> tuple[Path, Path]:
    """Cache genuine COG patches with provenance. Never label them as lithium ground truth."""
    destination.mkdir(parents=True, exist_ok=True)
    cfg = _sentinel_config(load_config("model_2"))
    records: list[dict] = []
    failures: list[dict] = []
    for year in years:
        provider = Sentinel2Provider(replace(cfg, datetime=f"{year}-01-01/{year}-12-31",
                                              search_limit=10))
        for name, (lat, lon) in locations.items():
            if len(records) >= max_patches:
                break
            identity = f"{name}_{year}"
            file = destination / f"{identity}.npy"
            try:
                path, meta = download_patch(identity, lat, lon, file, provider)
                scene = provider.search_best_scene(lat, lon)
                quality = read_scene_quality(scene, lat, lon, cfg.patch_size_m, cfg.patch_pixels)
                pixels = np.load(path, mmap_mode="r")
                finite_fraction = float(np.isfinite(pixels).mean())
                status = quality["quality_status"]
                if finite_fraction < .7:
                    status = "REJECT_NODATA"
                records.append({"image_id":identity, "year":year, "latitude":lat,
                    "longitude":lon, "path":str(path.resolve()), "scene_id":meta.get("scene_id"),
                    "scene_datetime":meta.get("datetime"),
                    "scene_cloud_cover_percent":meta.get("cloud_cover"),
                    "local_cloud_fraction":quality["local_cloud_fraction"],
                    "finite_fraction":finite_fraction, "status":status})
                print(f"PATCH {identity}: {status} scene={meta.get('scene_id')}", flush=True)
            except Exception as exc:
                failures.append({"image_id":identity, "error_type":type(exc).__name__, "error":str(exc)[:500]})
                print(f"PATCH {identity}: FAILED {str(exc)[:200]}", flush=True)
    manifest_path = destination / "manifest.csv"
    failures_path = destination / "failures.csv"
    pd.DataFrame(records, columns=[
        "image_id","year","latitude","longitude","path","scene_id","scene_datetime",
        "scene_cloud_cover_percent","local_cloud_fraction","finite_fraction","status",
    ]).to_csv(manifest_path,index=False)
    pd.DataFrame(failures, columns=["image_id","error_type","error"]).to_csv(failures_path,index=False)
    (destination/"provenance.json").write_text(json.dumps({
        "provider":cfg.stac_url, "collection":cfg.collection,"bands":list(cfg.bands),
        "patch_m":cfg.patch_size_m,"pixels":cfg.patch_pixels,"years":list(years),
        "note":"Unlabelled remote-sensing imagery; never score scientific accuracy from these patches.",
    },indent=2),encoding="utf-8")
    return manifest_path, failures_path
