"""Predeclared physically interpretable Sentinel-2 spectral feature policies.

Only selected inside a training fold. Never refit feature-selection statistics
on validation imagery, and never modify the original 126-feature cache.
"""
from __future__ import annotations

BANDS = ("red", "green", "blue", "nir", "swir16", "swir22")
STATS = ("mean", "std", "p10", "p50", "p90", "iqr",
         "texture_mean", "texture_p90")


def select_spectral_features(available: list[str], policy: str) -> list[str]:
    """Return the declared feature subset in stable training-schema order."""
    if policy == "full_126":
        if len(available) != 126 or len(set(available)) != len(available):
            raise ValueError("Full Model 2 spectral schema must have 126 unique features")
        return list(available)
    if policy != "direct_band_48":
        raise ValueError(f"Unsupported spectral policy: {policy}")
    expected = {f"{band}_{stat}" for band in BANDS for stat in STATS}
    selected = [name for name in available if name in expected]
    if len(selected) != 48 or len(set(selected)) != 48:
        missing = sorted(expected - set(selected))
        raise ValueError(f"Invalid spectral band summary schema; missing {missing}")
    return selected
