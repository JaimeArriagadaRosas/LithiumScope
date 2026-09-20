from __future__ import annotations

import numpy as np

FEATURE_EXTRACTOR_VERSION = 3
DEFAULT_BAND_NAMES = (
    "red",
    "green",
    "blue",
    "nir",
    "swir16",
    "swir22",
)


def _summary(prefix: str, values: np.ndarray) -> dict[str, float]:
    finite = np.asarray(values, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_p10": 0.0,
            f"{prefix}_p50": 0.0,
            f"{prefix}_p90": 0.0,
            f"{prefix}_iqr": 0.0,
        }

    p10, p25, p50, p75, p90 = np.percentile(
        finite,
        [10, 25, 50, 75, 90],
    )
    return {
        f"{prefix}_mean": float(np.mean(finite)),
        f"{prefix}_std": float(np.std(finite)),
        f"{prefix}_p10": float(p10),
        f"{prefix}_p50": float(p50),
        f"{prefix}_p90": float(p90),
        f"{prefix}_iqr": float(p75 - p25),
    }


def _texture_summary(
    prefix: str,
    values: np.ndarray,
) -> dict[str, float]:
    array = np.asarray(
        values,
        dtype=np.float32,
    )
    if array.ndim != 2:
        return {
            f"{prefix}_texture_mean": 0.0,
            f"{prefix}_texture_p90": 0.0,
        }

    gradients = []
    if array.shape[0] > 1:
        gradients.append(
            np.abs(
                np.diff(
                    array,
                    axis=0,
                )
            ).ravel()
        )
    if array.shape[1] > 1:
        gradients.append(
            np.abs(
                np.diff(
                    array,
                    axis=1,
                )
            ).ravel()
        )
    if not gradients:
        return {
            f"{prefix}_texture_mean": 0.0,
            f"{prefix}_texture_p90": 0.0,
        }

    finite = np.concatenate(
        gradients
    )
    finite = finite[
        np.isfinite(finite)
    ]
    if not finite.size:
        return {
            f"{prefix}_texture_mean": 0.0,
            f"{prefix}_texture_p90": 0.0,
        }
    return {
        f"{prefix}_texture_mean": float(
            np.mean(finite)
        ),
        f"{prefix}_texture_p90": float(
            np.percentile(
                finite,
                90,
            )
        ),
    }


def _normalized_difference(
    first: np.ndarray,
    second: np.ndarray,
) -> np.ndarray:
    denominator = first + second
    return np.divide(
        first - second,
        denominator,
        out=np.zeros_like(first, dtype=np.float32),
        where=np.abs(denominator) > 1e-6,
    )


def _safe_ratio(
    numerator: np.ndarray,
    denominator: np.ndarray,
) -> np.ndarray:
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float32),
        where=np.abs(denominator) > 1e-6,
    )


def extract_spectral_features(
    array: np.ndarray,
    band_names: tuple[str, ...] | list[str] = DEFAULT_BAND_NAMES,
) -> dict[str, float]:
    """Extract raw reflectance summaries plus physically interpretable ratios."""
    image = np.asarray(array, dtype=np.float32)
    names = tuple(str(name) for name in band_names)
    if image.ndim != 3:
        raise ValueError(
            f"Expected [bands, height, width] imagery, got {image.shape}."
        )
    if image.shape[0] != len(names):
        raise ValueError(
            "Band count does not match configured Sentinel schema: "
            f"image={image.shape[0]} configured={len(names)}."
        )

    bands = {
        name: image[index]
        for index, name in enumerate(names)
    }
    features: dict[str, float] = {}

    for name, band in bands.items():
        features.update(
            _summary(
                name,
                band,
            )
        )
        features.update(
            _texture_summary(
                name,
                band,
            )
        )

    required = set(DEFAULT_BAND_NAMES)
    if required.issubset(bands):
        ndvi = _normalized_difference(
            bands["nir"],
            bands["red"],
        )
        swir_plus_red = (
            bands["swir16"]
            + bands["red"]
        )
        nir_plus_blue = (
            bands["nir"]
            + bands["blue"]
        )
        bsi = _normalized_difference(
            swir_plus_red,
            nir_plus_blue,
        )
        derived = {
            "ndvi": ndvi,
            "ndmi": _normalized_difference(bands["nir"], bands["swir16"]),
            "nbr": _normalized_difference(bands["nir"], bands["swir22"]),
            "nbr2": _normalized_difference(
                bands["swir16"],
                bands["swir22"],
            ),
            "mndwi": _normalized_difference(
                bands["green"],
                bands["swir16"],
            ),
            "bare_soil_index": bsi,
            "dry_bare_soil_index": (
                _normalized_difference(
                    bands["swir16"],
                    bands["green"],
                )
                - ndvi
            ),
            "nir_swir22_nd": _normalized_difference(
                bands["nir"],
                bands["swir22"],
            ),
            "red_swir16_nd": _normalized_difference(
                bands["red"],
                bands["swir16"],
            ),
            "swir16_swir22_ratio": _safe_ratio(
                bands["swir16"],
                bands["swir22"],
            ),
            "nir_swir16_ratio": _safe_ratio(
                bands["nir"],
                bands["swir16"],
            ),
            "red_blue_ratio": _safe_ratio(
                bands["red"],
                bands["blue"],
            ),
            "green_red_ratio": _safe_ratio(
                bands["green"],
                bands["red"],
            ),
        }
        for name, values in derived.items():
            features.update(_summary(name, values))

    return features
