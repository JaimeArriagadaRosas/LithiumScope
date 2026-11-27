"""Strict new-cohort Sentinel patches, co-registered and SCL-masked.

Legacy frozen assets and their historical reader remain untouched.
"""
from __future__ import annotations

from datetime import date
import json
from uuid import uuid4

import numpy as np

from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.reproducibility import canonical_json_hash
from lithiumscope.datasets.adapters.sentinel2 import validate_metric_raster
from lithiumscope.model_2.methodology.spectral.satellite_quality import (
    ScientificScenePolicy, analyze_arrays, scene_metadata, BLOCKING_CLASSES, SPECTRAL_BANDS,
)
from lithiumscope.tools.acquisition.sentinel2.provider import (
    SentinelAssetError, SentinelSceneUnavailable, _asset_href,
)

POLICY = ScientificScenePolicy(min_joint_valid=.95, max_scl_blocked=.05, min_not_vegetated=0.)


def validate_interval(interval):
    """New data acquisition must have a closed, predeclared date interval."""
    try:
        start, end = (date.fromisoformat(part) for part in interval.split("/"))
    except (ValueError, AttributeError) as exc:
        raise ValueError("Scientific acquisition requires YYYY-MM-DD/YYYY-MM-DD") from exc
    if start > end:
        raise ValueError("Acquisition interval is reversed")
    return start, end


def read_registered_scene(scene, latitude, longitude):
    """Use one exact projected 20-m grid and verify native asset grid compatibility."""
    import rasterio
    from rasterio.transform import from_origin, array_bounds
    from rasterio.vrt import WarpedVRT
    from rasterio.enums import Resampling
    from rasterio.warp import transform

    radiometry = scene_metadata(scene, SPECTRAL_BANDS)
    for name in SPECTRAL_BANDS:
        entry = scene["assets"][name]["raster:bands"][0]
        if "nodata" not in entry:
            raise ValueError("Missing STAC nodata semantics")
    if not scene.get("id") or not scene.get("properties", {}).get("datetime"):
        raise ValueError("Missing scene identity or acquisition date")
    arrays, evidence = [], {}
    grid = None
    with rasterio.Env(AWS_NO_SIGN_REQUEST="YES", GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                      GDAL_HTTP_MAX_RETRY="3", GDAL_HTTP_TIMEOUT="60"):
        for band in (*SPECTRAL_BANDS, "scl"):
            with rasterio.open(_asset_href(scene, band)) as ds:
                native = validate_metric_raster(ds)
                resolution = np.array([native.a, -native.e])
                if not np.allclose(resolution, resolution[0]) or not any(
                        np.isclose(resolution[0], value) for value in (10., 20., 60.)):
                    raise ValueError("Unsupported native Sentinel-2 grid resolution")
                if grid is None:
                    xx, yy = transform("EPSG:4326", ds.crs, [longitude], [latitude])
                    affine = from_origin(xx[0]-320., yy[0]+320., 20., 20.)
                    grid = (ds.crs, affine, native)
                crs, affine, reference = grid
                if ds.crs != crs:
                    raise ValueError("Asset CRS differs from reference band")
                alignment = min(native.a, reference.a)
                offsets = np.array([native.c-reference.c, native.f-reference.f])/alignment
                if not np.allclose(offsets, np.rint(offsets), rtol=0, atol=1e-5):
                    raise ValueError("Native asset grids are displaced")
                left, bottom, right, top = array_bounds(32, 32, affine)
                bounds = ds.bounds
                if (bounds.left > left+1e-5 or bounds.bottom > bottom+1e-5
                        or bounds.right < right-1e-5 or bounds.top < top-1e-5):
                    raise ValueError("Asset does not cover the complete patch footprint")
                categorical = band == "scl"
                declared = ds.nodata if categorical else radiometry[band]["nodata"]
                if ds.nodata is not None and declared is not None and not np.isclose(ds.nodata, float(declared), equal_nan=True):
                    raise ValueError("STAC and raster nodata conventions disagree")
                # Warp only to this exact target grid. Preserve nodata rather than filling it.
                options = {"crs": crs, "transform": affine, "width": 32, "height": 32,
                           "dtype": "float32", "nodata": np.nan,
                           "resampling": Resampling.nearest if categorical else Resampling.bilinear}
                if declared is not None:
                    options["src_nodata"] = float(declared)
                with WarpedVRT(ds, **options) as vrt:
                    value = vrt.read(1, masked=True).astype(np.float32).filled(np.nan)
                arrays.append(value)
                evidence[band] = {"native_transform": list(native)[:6], "native_resolution_m": resolution.tolist(),
                                  "source_crs": ds.crs.to_string(),
                                  "resampling": "nearest" if categorical else "bilinear"}
    raw, scl = np.stack(arrays[:-1]), arrays[-1]
    qa = analyze_arrays(raw, scl, policy=POLICY)
    valid_scl = np.isfinite(scl) & np.equal(scl, np.rint(scl)) & (scl >= 0) & (scl <= 11)
    usable = valid_scl & ~np.isin(scl, list(BLOCKING_CLASSES)) & np.isfinite(raw).all(axis=0)
    qa["joint_usable_fraction"] = float(usable.mean())
    if qa["joint_usable_fraction"] < POLICY.min_joint_valid:
        qa["rejection_reasons"].append("INSUFFICIENT_JOINT_USABLE_PIXELS")
        qa["scientifically_accepted_proxy"] = False
    masked = np.where(usable[None, :, :], raw, np.nan).astype(np.float32)
    # Keep raw DN, not reflectance. The next versioned extractor owns conversion.
    return masked, scl, qa, {"crs": grid[0].to_string(), "transform": list(grid[1])[:6],
                             "bounds": list(array_bounds(32, 32, grid[1])), "assets": evidence,
                             "radiometry": radiometry}


def _atomic_npy(path, array):
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as stream:
            np.save(stream, array, allow_pickle=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path, payload):
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def verify_registered_mask(path, meta):
    """Verify sidecar, mask bytes and actual pixel support of newly acquired data."""
    mask_path = path.with_suffix(".scl.npy")
    if meta.get("acquisition_geometry") != "registered_metric_grid_v2" or not mask_path.is_file():
        raise ValueError("Registered grid and SCL evidence required")
    if meta.get("scl_sha256") != file_sha256(mask_path):
        raise ValueError("SCL mask hash mismatch")
    scl = np.load(mask_path, allow_pickle=False)
    raw = np.load(path, allow_pickle=False)
    if scl.shape != (32, 32) or raw.shape != (6, 32, 32):
        raise ValueError("Registered array shape mismatch")
    valid = np.isfinite(scl) & np.equal(scl, np.rint(scl)) & (scl >= 0) & (scl <= 11)
    valid &= ~np.isin(scl, list(BLOCKING_CLASSES))
    joint = np.isfinite(raw).all(axis=0)
    if np.any(np.isfinite(raw[:, ~valid])) or float(joint.mean()) < POLICY.min_joint_valid:
        raise ValueError("SCL mask and acquired image disagree")
    if not meta.get("scientific_patch_qa", {}).get("scientifically_accepted_proxy"):
        raise ValueError("Missing successful local QA")
    return scl


def download_registered_patch(sample_id, latitude, longitude, destination, provider, signature):
    """Select the best qualifying local patch and atomically publish its evidence."""
    config = provider.config
    start, end = validate_interval(config.datetime)
    if tuple(config.bands) != SPECTRAL_BANDS or config.patch_pixels != 32 or config.patch_size_m != 640:
        raise ValueError("Registered acquisition requires canonical six-band 640m/32px grid")
    meta_path = destination.with_suffix(".json")
    if destination.exists() or meta_path.exists() or destination.with_suffix(".scl.npy").exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if (meta["sample_id"] != sample_id or meta["acquisition_signature"] != signature
                    or meta["image_sha256"] != file_sha256(destination)
                    or not np.allclose([meta["latitude"], meta["longitude"]], [latitude, longitude],
                                       rtol=0, atol=1e-8)):
                raise ValueError("Cached registered patch identity differs")
            verify_registered_mask(destination, meta)
            return destination, meta
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise SentinelAssetError("Invalid registered cache; preserve for explicit recovery") from exc
    candidates = provider.search_candidate_scenes(latitude, longitude, datetime=config.datetime,
                                                   max_items=min(config.search_limit, POLICY.max_candidates))
    audits, winner = [], None
    for scene in candidates:
        sid = str(scene.get("id", ""))
        try:
            properties = scene["properties"]
            day = date.fromisoformat(str(properties["datetime"])[:10])
            cloud = float(properties["eo:cloud_cover"])
            if (scene.get("collection") != config.collection or not start <= day <= end
                    or not np.isfinite(cloud) or not 0 <= cloud <= config.cloud_cover_max):
                raise ValueError("Scene outside registered interval or cloud policy")
            raw, scl, qa, geometry = read_registered_scene(scene, latitude, longitude)
            audits.append({"scene_id": sid, "date": day.isoformat(), **qa})
            if qa["scientifically_accepted_proxy"]:
                rank = (-qa["joint_usable_fraction"], cloud, day, sid)
                if winner is None or rank < winner[0]:
                    winner = rank, scene, raw, scl, qa, geometry
        except (OSError, ValueError, KeyError, SentinelAssetError) as exc:
            audits.append({"scene_id": sid, "scientifically_accepted_proxy": False,
                           "rejection_reasons": [type(exc).__name__ + ": " + str(exc)[:200]]})
    destination.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(destination.with_suffix(".scene-audit.json"), audits)
    if winner is None:
        raise SentinelSceneUnavailable("No scene passes registered local QA; see scene audit")
    _, scene, raw, scl, qa, geometry = winner
    mask_path = destination.with_suffix(".scl.npy")
    _atomic_npy(destination, raw)
    _atomic_npy(mask_path, scl)
    meta = {"sample_id": sample_id, "latitude": latitude, "longitude": longitude,
            "scene_id": scene["id"], "datetime": scene["properties"]["datetime"],
            "cloud_cover": scene["properties"]["eo:cloud_cover"], "bands": list(config.bands),
            "collection": scene["collection"], "patch_pixels": 32, "patch_size_m": 640, "pixel_size_m": 20,
            "radiometric_representation": "raw_dn", "image_sha256": file_sha256(destination),
            "scl_sha256": file_sha256(mask_path), "acquisition_signature": signature,
            "patch_identity": canonical_json_hash({"scene": scene["id"], "signature": signature,
                                                    "sample": sample_id, "lat": latitude, "lon": longitude}),
            "acquisition_geometry": "registered_metric_grid_v2", "geometry": geometry,
            "scientific_patch_qa": qa}
    _atomic_json(meta_path, meta)
    return destination, meta
