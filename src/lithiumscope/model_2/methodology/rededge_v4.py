"""Versioned M2 red-edge additions. The historical 126-D block is immutable.

Four 20-m Sentinel-2 L2A assets (B05/B06/B07/B8A) become 32 band
descriptors and 12 red-edge index descriptors; zero pseudo-Li features.
"""

from __future__ import annotations

import numpy as np

from lithiumscope.model_2.steps.step_05_spectral_features import (
    _summary,
    _texture_summary,
)

VERSION = 4
NEW_BANDS = ("rededge1", "rededge2", "rededge3", "nir08")
NEW_FEATURES = 44
TOTAL_FEATURES = 170


def reflectance_from_dn(dn: np.ndarray, scales: tuple[tuple[float, float], ...]) -> np.ndarray:
    """Apply STAC raster:bands scale + offset individually, preserving nodata."""
    data = np.asarray(dn, dtype=np.float32)
    if data.ndim != 3 or data.shape[0] != 4 or len(scales) != 4:
        raise ValueError("Expected four red-edge bands CxHxW with four STAC scales")
    result = np.full_like(data, np.nan)
    for i, (scale, offset) in enumerate(scales):
        if not np.isfinite(scale) or scale <= 0 or not np.isfinite(offset):
            raise ValueError("Invalid radiometric scale or offset")
        valid = np.isfinite(data[i])
        result[i, valid] = data[i, valid] * scale + offset
    return result


def _nd(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    result = np.full_like(first, np.nan, dtype=np.float32)
    den = first + second
    mask = np.isfinite(first) & np.isfinite(second) & (np.abs(den) > 1e-6)
    np.divide(first - second, den, out=result, where=mask)
    return result


def extract_rededge_v4(
    reflectance: np.ndarray,
    *,
    min_joint_valid: float = 0.5,
) -> tuple[dict[str, float], float]:
    """Extract exactly 44 features from four physical reflectance bands.

    Fails if coverage is too low. Quality policy for TRAIN remains independent
    of and strictly local to outer TRAIN. TEST never changes labels or groups.
    """
    cube = np.asarray(reflectance, dtype=np.float32)
    if cube.ndim != 3 or cube.shape[0] != 4 or min(cube.shape[1:]) < 2:
        raise ValueError("Red-edge v4 expects four co-registered 2-D bands")
    if not 0 < min_joint_valid <= 1:
        raise ValueError("Invalid joint coverage cutoff")
    valid = np.isfinite(cube).all(axis=0)
    coverage = float(np.mean(valid))
    if coverage < min_joint_valid:
        raise ValueError(f"Red-edge joint coverage {coverage:.2%} below cutoff")
    arrays = {name: np.where(valid, cube[i], np.nan) for i, name in enumerate(NEW_BANDS)}
    features = {}
    for name, array in arrays.items():
        features.update(_summary(f"{name}_refl", array))
        features.update(_texture_summary(f"{name}_refl", array))
    indices = {
        "ndre1_refl": _nd(arrays["nir08"], arrays["rededge1"]),
        "ndre2_refl": _nd(arrays["rededge3"], arrays["rededge2"]),
    }
    for name, values in indices.items():
        features.update(_summary(name, values))
    if len(features) != NEW_FEATURES or len(set(features)) != NEW_FEATURES:
        raise ValueError("Red-edge schema changed")
    if not np.isfinite(np.array(list(features.values()), float)).all():
        raise ValueError("Non-finite red-edge spectral descriptor")
    return features, coverage


def append_rededge_features(
    baseline: dict[str, float],
    rededge: dict[str, float],
) -> dict[str, float]:
    """Preserve the original 126-D serialized values and column order."""
    if len(baseline) != 126 or len(rededge) != NEW_FEATURES:
        raise ValueError("Wrong M2 spectral feature block dimensions")
    if set(baseline) & set(rededge):
        raise ValueError("New red-edge features overlap legacy names")
    result = dict(baseline)
    result.update(rededge)
    if len(result) != TOTAL_FEATURES:
        raise ValueError("M2 v4 dimension changed")
    return result
