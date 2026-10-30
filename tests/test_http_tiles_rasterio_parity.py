"""Numerical regression against rasterio on actual tiled GeoTIFF files."""
from __future__ import annotations

import numpy as np
import pytest

from lithiumscope.experiments.http_tiles import extract_http_patch
from lithiumscope.model_2.data.sentinel2 import _read_patch


@pytest.mark.parametrize("resolution", [10, 20])
def test_http_bilinear_matches_rasterio_on_local_geotiff(tmp_path, resolution):
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.warp import transform
    from rasterio.enums import Resampling

    width = 256
    # Linear gradients are exact under bilinear resampling in non-boundary regions.
    yy, xx = np.mgrid[:width, :width]
    values = (3.5 * xx + 7.2 * yy + 19).astype("float32")
    transform_affine = from_origin(500000, 7500000, resolution, resolution)
    file_path = tmp_path / f"synthetic_{resolution}m.tif"
    with rasterio.open(
        file_path, "w", driver="GTiff", width=width, height=width, count=1,
        dtype="float32", crs="EPSG:32719", transform=transform_affine,
        tiled=True, blockxsize=64, blockysize=64, compress="deflate",
        nodata=-9999,
    ) as dst:
        dst.write(values, 1)
    mid_x, mid_y = 500000 + 100 * resolution, 7500000 - 120 * resolution
    longitudes, latitudes = transform("EPSG:32719", "EPSG:4326", [mid_x], [mid_y])
    lat, lon = latitudes[0], longitudes[0]
    actual = extract_http_patch(str(file_path), lat, lon, 640, 32, resampling="bilinear")
    reference = _read_patch(str(file_path), lat, lon, 640, 32)
    assert actual.shape == (32, 32)
    np.testing.assert_allclose(actual, reference, rtol=1e-5, atol=2e-3)


def test_http_nearest_matches_rasterio_on_local_geotiff(tmp_path):
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.warp import transform
    from rasterio.windows import from_bounds
    from rasterio.enums import Resampling

    yy, xx = np.mgrid[:256, :256]
    values = ((xx // 25 + yy // 30) % 12).astype("uint8")
    affine = from_origin(500000, 7500000, 20, 20)
    file_path = tmp_path / "classes.tif"
    with rasterio.open(
        file_path,"w",driver="GTiff",width=256,height=256,count=1,
        dtype="uint8",crs="EPSG:32719",transform=affine,
        tiled=True,blockxsize=64,blockysize=64,compress="deflate",
        nodata=0
    ) as ds:
        ds.write(values,1)
    x, y = 500000+125*20,7500000-127*20
    lon, lat = transform("EPSG:32719","EPSG:4326",[x],[y])
    actual = extract_http_patch(str(file_path),lat[0],lon[0],640,32,resampling="nearest")
    with rasterio.open(file_path) as ds:
        xs,ys=transform("EPSG:4326",ds.crs,[lon[0]],[lat[0]])
        window=from_bounds(xs[0]-320,ys[0]-320,xs[0]+320,ys[0]+320,ds.transform)
        expected=ds.read(1,window=window,out_shape=(32,32),masked=True,
                         boundless=True,resampling=Resampling.nearest).astype(np.float32).filled(np.nan)
    np.testing.assert_allclose(actual,expected,equal_nan=True)
