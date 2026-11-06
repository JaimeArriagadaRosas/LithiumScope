"""Sentinel-2 evaluation patches with per-asset timeouts and resumable manifests."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np
import pandas as pd

from lithiumscope.core.config import load_config
from lithiumscope.model_2.data.builder import _sentinel_config
from lithiumscope.model_2.data.sentinel2 import Sentinel2Provider, _asset_href

PILOT_LOCATIONS = {
    "atacama": (-23.10, -68.18),
    "tara": (-23.08, -67.33),
    "maricunga": (-26.91, -69.08),
    "pedernales": (-26.25, -68.60),
}
YEARS = (2024, 2025)
CLOUD_OR_INVALID_SCL = (0, 1, 3, 7, 8, 9, 10, 11)
COLUMNS = ("image_id","year","latitude","longitude","path","scene_id",
           "scene_datetime","scene_cloud_cover_percent","local_cloud_fraction",
           "finite_fraction","status")


class AssetTimeoutError(RuntimeError):
    """A COG asset did not complete within its bounded execution window."""


def read_asset_isolated(href: str, latitude: float, longitude: float, size_m: float,
                        pixels: int, *, mode: str = "reflectance",
                        timeout_seconds: int = 45) -> np.ndarray:
    """Isolate GDAL networking so a single blocked URL cannot stall the entire batch."""
    with tempfile.TemporaryDirectory(prefix="lithiumscope_s2_") as tmp:
        output = Path(tmp) / "asset.npy"
        command = [sys.executable, "-m", "lithiumscope.experiments.acquisition.raster_worker",
                   "--href", href, "--lat", str(latitude), "--lon", str(longitude),
                   "--size", str(size_m), "--pixels", str(pixels), "--mode", mode,
                   "--output", str(output)]
        try:
            finished = subprocess.run(command, capture_output=True, text=True,
                                      timeout=timeout_seconds, check=False)
        except subprocess.TimeoutExpired as exc:
            raise AssetTimeoutError(f"{mode} raster read exceeded {timeout_seconds}s") from exc
        if finished.returncode != 0:
            detail = (finished.stderr or finished.stdout or "").strip()
            raise RuntimeError(f"{mode} raster read failed: {detail[-600:]}")
        if not output.is_file():
            raise RuntimeError(f"{mode} raster worker returned no data")
        array = np.load(output, allow_pickle=False)
        if array.shape != (pixels, pixels):
            raise ValueError(f"Unexpected band shape {array.shape}")
        return array


def quality_from_scl(scl: np.ndarray) -> dict:
    if scl.ndim != 2:
        raise ValueError("SCL must be a 2D class raster")
    invalid = np.isin(scl, CLOUD_OR_INVALID_SCL) | ~np.isfinite(scl)
    fraction = float(np.mean(invalid))
    return {"quality_status": "ACCEPT" if fraction <= .35 else "REJECT_CLOUD",
            "local_cloud_fraction": fraction}


def acquire_pilot(destination: Path, *, years=YEARS, locations=PILOT_LOCATIONS,
                  max_patches: int = 40, timeout_seconds: int = 45,
                  date_window: tuple[str, str] | None = None) -> tuple[Path, Path]:
    """Download genuine scenes, checking SCL before retrieving six spectral bands."""
    destination.mkdir(parents=True, exist_ok=True)
    cfg = _sentinel_config(load_config("model_2"))
    manifest = destination / "manifest.csv"
    failures_file = destination / "failures.csv"
    existing = pd.read_csv(manifest) if manifest.is_file() else pd.DataFrame(columns=COLUMNS)
    records = {str(row["image_id"]): row for row in existing.to_dict("records")}
    failures = pd.read_csv(failures_file).to_dict("records") if failures_file.is_file() else []
    attempts = 0
    for year in years:
        window = (f"{year}-{date_window[0]}/{year}-{date_window[1]}" if date_window
                  else f"{year}-01-01/{year}-12-31")
        provider = Sentinel2Provider(replace(cfg, datetime=window, search_limit=8))
        for name, (lat, lon) in locations.items():
            if attempts >= max_patches:
                break
            attempts += 1
            identity = f"{name}_{year}"
            output = destination / f"{identity}.npy"
            if identity in records and output.is_file():
                print(f"PATCH {identity}: CACHED", flush=True)
                continue
            print(f"PATCH {identity}: searching...", flush=True)
            try:
                scene = provider.search_best_scene(lat, lon)
                if not scene.get("assets", {}).get("scl"):
                    raise ValueError("Scene missing SCL quality layer")
                scl = read_asset_isolated(_asset_href(scene,"scl"),lat,lon,cfg.patch_size_m,
                                          cfg.patch_pixels,mode="scl",timeout_seconds=timeout_seconds)
                quality = quality_from_scl(scl)
                if quality["quality_status"] != "ACCEPT":
                    print(f"PATCH {identity}: rejected cloudy SCL",flush=True)
                    continue
                arrays = []
                for band in cfg.bands:
                    started=time.monotonic()
                    print(f"PATCH {identity}: reading {band}...",flush=True)
                    array = read_asset_isolated(_asset_href(scene,band),lat,lon,
                                                cfg.patch_size_m,cfg.patch_pixels,
                                                timeout_seconds=timeout_seconds)
                    arrays.append(array)
                    print(f"PATCH {identity}: {band} read in {time.monotonic()-started:.1f}s",
                          flush=True)
                stack = np.stack(arrays).astype(np.float32)
                finite_fraction = float(np.isfinite(stack).mean())
                status = "ACCEPT" if finite_fraction >= .7 else "REJECT_NODATA"
                if status == "ACCEPT":
                    # Saving only complete accepted patches prevents partial cache poisoning.
                    temp = output.with_suffix(".part.npy")
                    np.save(temp,stack)
                    temp.replace(output)
                    meta = {"image_id":identity,"scene_id":scene.get("id"),
                            "datetime":scene.get("properties",{}).get("datetime"),
                            "bands":list(cfg.bands),"local_cloud_fraction":quality["local_cloud_fraction"]}
                    output.with_suffix(".json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
                records[identity] = {"image_id":identity,"year":year,"latitude":lat,
                    "longitude":lon,"path":str(output.resolve()) if status=="ACCEPT" else "",
                    "scene_id":scene.get("id"),"scene_datetime":scene.get("properties",{}).get("datetime"),
                    "scene_cloud_cover_percent":scene.get("properties",{}).get("eo:cloud_cover"),
                    "local_cloud_fraction":quality["local_cloud_fraction"],
                    "finite_fraction":finite_fraction,"status":status}
                print(f"PATCH {identity}: {status}",flush=True)
            except Exception as exc:
                failures.append({"image_id":identity,"error_type":type(exc).__name__,
                                 "error":str(exc)[:500]})
                print(f"PATCH {identity}: FAILED {type(exc).__name__} {str(exc)[:180]}",flush=True)
            finally:
                pd.DataFrame(list(records.values()),columns=COLUMNS).to_csv(manifest,index=False)
                pd.DataFrame(failures,columns=["image_id","error_type","error"]).to_csv(
                    failures_file,index=False)
    (destination/"provenance.json").write_text(json.dumps({
        "provider":cfg.stac_url,"collection":cfg.collection,"bands":list(cfg.bands),
        "patch_m":cfg.patch_size_m,"pixels":cfg.patch_pixels,"years":list(years),
        "timeout_seconds_per_asset":timeout_seconds,
        "note":"Unlabelled satellite imagery; M2 scores do not prove lithium deposits.",
    },indent=2),encoding="utf-8")
    return manifest, failures_file
