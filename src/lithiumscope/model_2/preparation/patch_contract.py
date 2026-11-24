"""Versioned patch geometry, semantic band and byte-identity contract for new candidates."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from lithiumscope.core.hashing import file_sha256
from lithiumscope.model_2.preparation.features.spectral import DEFAULT_BAND_NAMES


def make_patch_contract(*, bands=DEFAULT_BAND_NAMES, pixels=32, size_m=640):
    bands=tuple(bands)
    if bands!=DEFAULT_BAND_NAMES or pixels!=32 or size_m!=640:
        raise ValueError("Canonical Sentinel-2 patch must use ordered six bands and 640m/32px")
    return {"version":1,"bands":list(bands),"patch_pixels":32,"patch_size_m":640,
            "pixel_size_m":20,"collection":"sentinel-2-l2a","radiometric_representation":"raw_dn"}


def metadata_path(path):
    candidates=(path.with_suffix(path.suffix+".json"),path.with_suffix(".json"))
    existing=[p for p in candidates if p.is_file()]
    if len(existing)!=1:
        raise ValueError("Exactly one unambiguous patch provenance sidecar is required")
    return existing[0]


def verify_metadata(path,contract):
    expected=make_patch_contract(bands=contract["bands"],pixels=contract["patch_pixels"],size_m=contract["patch_size_m"])
    if contract!=expected:
        raise ValueError("Unrecognized patch contract")
    sidecar=metadata_path(path)
    meta=json.loads(sidecar.read_text(encoding="utf-8"))
    for key in ("bands","patch_pixels","patch_size_m","pixel_size_m","collection","radiometric_representation"):
        if meta.get(key)!=expected[key]:
            raise ValueError(f"Patch provenance mismatch: {key}")
    if meta.get("image_sha256")!=file_sha256(path):
        raise ValueError("Patch bytes do not match the recorded SHA256")
    if not isinstance(meta.get("scene_id"),str) or not meta["scene_id"].strip():
        raise ValueError("Missing patch scene identity")
    lat,lon=float(meta["latitude"]),float(meta["longitude"])
    if not (-90<=lat<=90 and -180<=lon<=180):
        raise ValueError("Invalid patch coordinates")
    return meta,sidecar


def validate_metric_raster(dataset):
    crs=dataset.crs
    if crs is None or not crs.is_projected:
        raise ValueError("Raster must use a projected metric CRS")
    _, factor=crs.linear_units_factor
    if not np.isclose(factor,1.,rtol=0,atol=1e-9):
        raise ValueError("Raster linear units must be metres")
    affine=dataset.transform
    if not np.isclose(affine.b,0,atol=1e-9) or not np.isclose(affine.d,0,atol=1e-9) or affine.a<=0 or affine.e>=0:
        raise ValueError("Raster must be north-up, with nonrotated positive pixel width")
    return affine


def load_verified_patch(path,contract):
    """Read NPY or GeoTIFF with no resampling and no value-changing imputation."""
    path=Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    meta,sidecar=verify_metadata(path,contract)
    if path.suffix.lower()==".npy":
        raw=np.load(path,allow_pickle=False)
        geometry="sidecar_declared_not_independently_georeferenced"
    elif path.suffix.lower() in {".tif",".tiff"}:
        import rasterio
        from rasterio.warp import transform
        with rasterio.open(path) as ds:
            affine=validate_metric_raster(ds)
            if ds.count!=6 or ds.height!=32 or ds.width!=32:
                raise ValueError("Unexpected GeoTIFF dimensions")
            if tuple(ds.descriptions)!=DEFAULT_BAND_NAMES:
                raise ValueError("GeoTIFF band descriptions must match model order")
            if not np.allclose([affine.a,-affine.e],[20.,20.],rtol=0,atol=1e-5):
                raise ValueError("GeoTIFF resolution differs from 20m patch")
            xs,ys=transform("EPSG:4326",ds.crs,[float(meta["longitude"])],[float(meta["latitude"])])
            cx=affine.c+affine.a*ds.width/2
            cy=affine.f+affine.e*ds.height/2
            if np.hypot(cx-xs[0],cy-ys[0])>10.:
                raise ValueError("GeoTIFF center disagrees with declared sample coordinates")
            raw=ds.read(masked=True).astype(np.float32).filled(np.nan)
        geometry="metric_crs_resolution_and_center_verified"
    else:
        raise ValueError("Expected local NPY or GeoTIFF patch")
    if np.asarray(raw).shape!=(6,32,32) or not np.issubdtype(raw.dtype,np.number) or np.iscomplexobj(raw):
        raise ValueError("Expected numeric six-band 32x32 patch")
    raw=np.asarray(raw,dtype=np.float32).copy()
    raw[~np.isfinite(raw)]=np.nan
    quality={"geometry_verification":geometry,"image_sha256":meta["image_sha256"],
             "sidecar_sha256":file_sha256(sidecar),"scene_id":meta["scene_id"],
             "joint_valid_fraction":float(np.isfinite(raw).all(axis=0).mean()),
             "band_valid_fractions":np.isfinite(raw).mean(axis=(1,2)).tolist(),
             "radiometric_representation":meta["radiometric_representation"]}
    return raw,meta,quality
