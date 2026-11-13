"""Experimental M2 Sentinel-2 gain perturbation, strictly train-only.

This is a low-amplitude robustness probe in stored DN space, NOT a physically
validated new satellite acquisition. Preserve band order, geometry, missingness
and parent label. Never write modified patches over source images.
"""
from __future__ import annotations

import hashlib

import numpy as np

from lithiumscope.model_2.steps.step_04_image_preprocessing import preprocess_image
from lithiumscope.model_2.steps.step_05_spectral_features import extract_spectral_features

BANDS = ("red", "green", "blue", "nir", "swir16", "swir22")


def seed_for_sample(sample_id: str, *, seed: int = 20261112) -> int:
    """Stable across filesystems/Python sessions; avoids Python hash randomization."""
    payload = f"{seed}:{sample_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def perturb_sentinel_dn(
    patch: np.ndarray,
    *,
    seed: int,
    common_gain_amplitude: float = 0.03,
    band_gain_amplitude: float = 0.007,
) -> np.ndarray:
    """Correlated sensor-amplitude jitter; no spatial shuffling or synthesis.

    Common gain is shared by all bands, with small per-band residual changes.
    No new information or independent training examples are created. Values
    remain DN, not BOA reflectance; only an exploratory augmentation.
    """
    array = np.asarray(patch, dtype=np.float32)
    if array.ndim != 3 or array.shape[0] != 6 or min(array.shape[1:]) < 2:
        raise ValueError("Expected six-band Sentinel-2 CxHxW patch")
    if not np.isfinite(array).any():
        raise ValueError("Empty spectral patch")
    if not (0 <= common_gain_amplitude <= 0.05):
        raise ValueError("Unsafe common DN gain amplitude")
    if not (0 <= band_gain_amplitude <= 0.015):
        raise ValueError("Unsafe band DN gain amplitude")
    rng = np.random.default_rng(seed)
    shared = rng.uniform(-common_gain_amplitude, common_gain_amplitude)
    residual = rng.uniform(-band_gain_amplitude, band_gain_amplitude, size=6)
    gains = (1.0 + shared) * (1.0 + residual)
    result = array * gains[:, None, None].astype(np.float32)
    if not np.array_equal(np.isfinite(result), np.isfinite(array)):
        raise ValueError("Spectral perturbation changed the valid-pixel mask")
    return result


def augmented_feature_record(patch: np.ndarray, *, sample_id: str) -> dict[str, float]:
    """Extract existing 126-D schema after small train-only raw-DN perturbation."""
    variant = perturb_sentinel_dn(patch, seed=seed_for_sample(sample_id))
    clean = preprocess_image(variant, normalize_per_band=False)
    record = extract_spectral_features(clean, band_names=BANDS)
    if len(record) != 126 or not np.isfinite(list(record.values())).all():
        raise ValueError("Augmentation drifted from 126-D spectral schema")
    return record
