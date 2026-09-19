from __future__ import annotations

import numpy as np


def preprocess_image(
    array: np.ndarray,
    *,
    normalize_per_band: bool = False,
) -> np.ndarray:
    """Clean imagery while preserving absolute inter-sample reflectance information."""
    result = np.asarray(array, dtype=np.float32).copy()
    result[~np.isfinite(result)] = np.nan

    for index in range(result.shape[0]):
        band = result[index]
        finite = band[np.isfinite(band)]
        fill = float(np.median(finite)) if finite.size else 0.0
        band = np.nan_to_num(
            band,
            nan=fill,
            posinf=fill,
            neginf=fill,
        )

        if normalize_per_band:
            mean = float(band.mean())
            std = float(band.std())
            if std > 0:
                band = (band - mean) / std

        result[index] = band

    return result
