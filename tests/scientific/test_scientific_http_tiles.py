"""Pixel-centre interpolation and categorical sampling invariants."""
import numpy as np
from unittest.mock import Mock

from lithiumscope.experiments.acquisition.http_tiles import _interpolate


def _raster(data):
    page = Mock()
    page.imagewidth = page.tilewidth = data.shape[1]
    page.imagelength = page.tilelength = data.shape[0]
    page.dataoffsets = [0]
    page.databytecounts = [1]
    page.nodata = None
    page.decode = lambda *_: (data.reshape(1, *data.shape, 1), None, None)
    source = Mock()
    source.read = lambda _: b"x"
    return page, source


def test_bilinear_weighted_average():
    page, source = _raster(np.array([[0, 10], [20, 30]], dtype=np.float32))
    result = _interpolate(page, source, np.array([0.5]), np.array([0.5]), "bilinear")
    assert np.isclose(result[0, 0], 15)


def test_categorical_nearest_preserves_class():
    page, source = _raster(np.array([[4, 5], [8, 9]], dtype=np.float32))
    result = _interpolate(page, source, np.array([0.9]), np.array([0.9]), "nearest")
    assert result[0, 0] == 9


def test_bilinear_nodata_exclusion():
    page, source = _raster(np.array([[np.nan, 10], [20, 30]], dtype=np.float32))
    result = _interpolate(page, source, np.array([0.5]), np.array([0.5]), "bilinear")
    assert np.isclose(result[0, 0], 20)
