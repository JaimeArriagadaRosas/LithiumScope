import numpy as np

from lithiumscope.model_2.steps.step_04_image_preprocessing import (
    preprocess_image,
)
from lithiumscope.model_2.steps.step_05_spectral_features import (
    extract_spectral_features,
)


def test_preprocessing_preserves_absolute_band_scale_by_default():
    image = np.stack(
        [
            np.full((4, 4), 100.0, dtype=np.float32),
            np.full((4, 4), 200.0, dtype=np.float32),
            np.full((4, 4), 300.0, dtype=np.float32),
            np.full((4, 4), 400.0, dtype=np.float32),
            np.full((4, 4), 500.0, dtype=np.float32),
            np.full((4, 4), 600.0, dtype=np.float32),
        ]
    )

    processed = preprocess_image(image)

    assert np.isclose(processed[0].mean(), 100.0)
    assert np.isclose(processed[-1].mean(), 600.0)


def test_spectral_features_include_raw_statistics_and_indices():
    red = np.full((3, 3), 2.0, dtype=np.float32)
    green = np.full((3, 3), 3.0, dtype=np.float32)
    blue = np.full((3, 3), 1.0, dtype=np.float32)
    nir = np.full((3, 3), 6.0, dtype=np.float32)
    swir16 = np.full((3, 3), 4.0, dtype=np.float32)
    swir22 = np.full((3, 3), 5.0, dtype=np.float32)
    image = np.stack([red, green, blue, nir, swir16, swir22])

    features = extract_spectral_features(
        image,
        band_names=(
            "red",
            "green",
            "blue",
            "nir",
            "swir16",
            "swir22",
        ),
    )

    assert np.isclose(features["red_mean"], 2.0)
    assert np.isclose(features["nir_mean"], 6.0)
    assert np.isclose(features["ndvi_mean"], 0.5)
    assert np.isclose(features["ndmi_mean"], 0.2)
    assert np.isclose(features["swir16_swir22_ratio_mean"], 0.8)
