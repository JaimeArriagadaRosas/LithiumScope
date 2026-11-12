"""Opt-in physically corrected, missingness-aware Sentinel-2 features.

The historical Model 2 126-D feature cache and model inference are not altered.
Use per-asset raster:bands scale + offset. Zero nodata was ALREADY masked in
rasterio, so only finite input pixel values are transformed. No broad clipping
to [0,1]: physically negative atmospherically-corrected reflectance is legal.

Since Sentinel-2 PB04 (2022), DN 1000 => 0 BOA with offset -0.1 and
scale 0.0001 in Earth Search COG STAC raster-band metadata.
"""
from __future__ import annotations

import numpy as np

from lithiumscope.model_2.data.sentinel2 import _masked_to_float
from lithiumscope.model_2.steps.step_05_spectral_features import (
    DEFAULT_BAND_NAMES,_summary,_texture_summary,
    extract_spectral_features,
)
from lithiumscope.model_2.steps.step_04_image_preprocessing import preprocess_image
from lithiumscope.model_2.methodology.satellite_quality import SPECTRAL_BANDS


def apply_radiometry(cube:np.ndarray,metadata:dict, *,
                     bands=SPECTRAL_BANDS)->np.ndarray:
    """Convert original masked DN to BOA reflectance with exact STAC scales."""
    raw=np.asarray(cube,dtype=np.float32)
    if raw.ndim!=3 or raw.shape[0]!=len(bands):
        raise ValueError("Expected CxHxW DN cube with specified band names")
    if set(metadata)!=set(bands):
        raise ValueError("Radiometry band metadata and model image schema differ")
    result=np.full_like(raw,np.nan,dtype=np.float32)
    for i,band in enumerate(bands):
        info=metadata[band]
        if not isinstance(info,dict) or "scale" not in info or "offset" not in info:
            raise ValueError(f"Missing scale/offset for {band}")
        scale=float(info["scale"]); offset=float(info["offset"])
        if not np.isfinite(scale) or scale<=0 or not np.isfinite(offset):
            raise ValueError("Invalid radiometric transform")
        valid=np.isfinite(raw[i])
        result[i][valid]=raw[i][valid]*scale+offset
    return result


def _safe_nd(a:np.ndarray,b:np.ndarray):
    denominator=a+b
    result=np.full_like(a,np.nan,dtype=np.float32)
    np.divide(a-b,denominator,out=result,
              where=np.isfinite(a)&np.isfinite(b)
              &np.isfinite(denominator)&(np.abs(denominator)>1e-6))
    return result


def _safe_div(a:np.ndarray,b:np.ndarray):
    result=np.full_like(a,np.nan,dtype=np.float32)
    np.divide(a,b,out=result,where=np.isfinite(a)&np.isfinite(b)
              &np.isfinite(b)&(np.abs(b)>1e-6))
    return result


def extract_valid_spectral_features(
    image:np.ndarray, *,
    bands=SPECTRAL_BANDS,
    min_finite_fraction:float=0.0,
)->dict[str,float]:
    """126 fixed-schema descriptors, using valid pixels only, no imputation.

    Excluding invalid pixels *before* ratio/index calculations prevents
    nonphysical fill values from contaminating derived statistics.
    """
    if not 0 <= min_finite_fraction <= 1:
        raise ValueError("Invalid coverage requirement")
    cube=np.asarray(image,dtype=np.float32)
    if cube.shape[0]!=len(bands) or cube.ndim!=3:
        raise ValueError("Invalid spectral tensor")
    joint=np.isfinite(cube).all(axis=0)
    if float(joint.mean()) < min_finite_fraction:
        raise ValueError("Joint valid-pixel coverage below required threshold")
    if not joint.any():
        raise ValueError("No simultaneously valid spectral pixels")
    # Enforce shared six-band validity to keep derived index supports consistent.
    bs={name:np.where(joint,cube[i],np.nan)
        for i,name in enumerate(bands)}
    features={}
    for name,band in bs.items():
        features.update(_summary(name,band))
        features.update(_texture_summary(name,band))
    if set(SPECTRAL_BANDS).issubset(bs):
        ndvi=_safe_nd(bs["nir"],bs["red"])
        derived={
            "ndvi":ndvi,
            "ndmi":_safe_nd(bs["nir"],bs["swir16"]),
            "nbr":_safe_nd(bs["nir"],bs["swir22"]),
            "nbr2":_safe_nd(bs["swir16"],bs["swir22"]),
            "mndwi":_safe_nd(bs["green"],bs["swir16"]),
            "bare_soil_index":_safe_nd(
                bs["swir16"]+bs["red"],bs["nir"]+bs["blue"]),
            "dry_bare_soil_index":_safe_nd(
                bs["swir16"],bs["green"])-ndvi,
            "nir_swir22_nd":_safe_nd(bs["nir"],bs["swir22"]),
            "red_swir16_nd":_safe_nd(bs["red"],bs["swir16"]),
            "swir16_swir22_ratio":_safe_div(bs["swir16"],bs["swir22"]),
            "nir_swir16_ratio":_safe_div(bs["nir"],bs["swir16"]),
            "red_blue_ratio":_safe_div(bs["red"],bs["blue"]),
            "green_red_ratio":_safe_div(bs["green"],bs["red"]),
        }
        for name,value in derived.items():
            features.update(_summary(name,value))
    expected=list(extract_spectral_features(
        np.ones_like(cube,dtype=np.float32),band_names=tuple(bands)))
    if list(features)!=expected or len(features)!=126:
        raise ValueError("Scientific and historical schemas do not align")
    if not np.isfinite(list(features.values())).all():
        raise ValueError("Physical feature computation produced nonfinite metrics")
    return features


def build_feature_variants(cube:np.ndarray,metadata:dict, *,
                           bands=SPECTRAL_BANDS)->dict[str,dict[str,float]]:
    """Three isolated variants to test missing-value and radiometry assumptions.

    Legacy cached 126-D baseline is handled separately and NEVER rewritten.
    """
    original=np.asarray(cube,dtype=np.float32)
    physical=apply_radiometry(original,metadata,bands=bands)
    # Mode 1: original DN without imputation; isolates median-fill effect.
    raw_valid=extract_valid_spectral_features(original,bands=bands)
    # Mode 2: physical reflectance and valid-only ratios.
    reflectance_valid=extract_valid_spectral_features(physical,bands=bands)
    # Mode 3: physical reflectance but old median-fill preprocessing (control).
    # Care: do NOT label this physically correct indices, as imputation alters
    # derived values. This is a matched control rather than the recommended method.
    reflectance_median=extract_spectral_features(
        preprocess_image(physical,normalize_per_band=False),band_names=tuple(bands))
    return {
        "raw_valid_only":raw_valid,
        "reflectance_valid_only":reflectance_valid,
        "reflectance_median_filled":reflectance_median,
    }
