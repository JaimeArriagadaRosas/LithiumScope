import numpy as np
import pytest
from lithiumscope.experiments.imagery import ImageScenario, image_catalog, perturb_image


def image():
    return np.arange(6*8*8,dtype=np.float32).reshape(6,8,8)


def test_catalog_deterministic():
    assert image_catalog() == image_catalog()
    assert len(image_catalog()) == 19


def test_baseline_preserves_original():
    original = image()
    transformed = perturb_image(original, ImageScenario("baseline","baseline",0))
    np.testing.assert_array_equal(original,transformed)
    assert transformed is not original


def test_missing_pixels_deterministic():
    scenario = ImageScenario("missing","missing_pixels",.5,31)
    np.testing.assert_array_equal(perturb_image(image(),scenario),
        perturb_image(image(),scenario),strict=True)


def test_missing_band():
    out = perturb_image(image(),ImageScenario("band","band_missing",2))
    assert np.isnan(out[2]).all()
    assert np.isfinite(out[1]).all()


def test_invalid_shape():
    with pytest.raises(ValueError):
        perturb_image(np.ones((5,5)),ImageScenario("baseline","baseline",0))
