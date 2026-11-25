"""Predeclared reflectance/texture arms; never select features using TEST."""
from __future__ import annotations

from lithiumscope.experiments.shared.paired_suite import Arm
from lithiumscope.model_2.preparation.features.rededge import NEW_BANDS
from lithiumscope.model_2.preparation.features.spectral import DEFAULT_BAND_NAMES

STATS = ("mean", "std", "p10", "p50", "p90", "iqr")
TEXTURE = ("texture_mean", "texture_p90")


def texture_arms(x, algorithms, mask, *, seed, mode, physical=False):
    """Compare ten-band reflectance alone to the same reflectance plus texture."""
    prefixes = (*DEFAULT_BAND_NAMES, *(f"{band}_refl" for band in NEW_BANDS))
    reflectance = [f"{band}_{stat}" for band in prefixes for stat in STATS]
    texture = [f"{band}_{stat}" for band in prefixes for stat in TEXTURE]
    if not x.columns.is_unique or not set(reflectance + texture).issubset(x.columns):
        raise ValueError("Ten-band reflectance and texture schema required")
    arms = []
    for algorithm in algorithms:
        ref = f"{algorithm}_reflectance"
        policy = {"seed": seed, "calibration": mode, "quality": "common_QA",
                  "extractor": "v5_physical" if physical else "v4_historical"}
        arms.extend([
            Arm(ref, algorithm, None, x.loc[:, reflectance],
                {**policy, "features": "reflectance_only"}, mask),
            Arm(f"{algorithm}_texture", algorithm, ref, x.loc[:, reflectance + texture],
                {**policy, "features": "reflectance_plus_texture"}, mask),
        ])
    return arms
