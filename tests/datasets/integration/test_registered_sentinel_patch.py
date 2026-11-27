import json
from pathlib import Path

import numpy as np
import pytest

from lithiumscope.tools.acquisition.sentinel2.registered_patch import (
    read_registered_scene, validate_interval, verify_registered_mask,
)
from lithiumscope.core.hashing import file_sha256


def test_registered_scene_common_grid_masks_cloud_and_rejects_displacement(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin
    from rasterio.warp import transform

    x, y = transform("EPSG:4326", "EPSG:32719", [-69.], [-20.])
    left, top = x[0]-640., y[0]+640.
    # Use common native 20m origin; sample grid is a 32px subset.
    names = ("red", "green", "blue", "nir", "swir16", "swir22", "scl")
    scene = {"id": "local", "collection": "sentinel-2-l2a",
             "properties": {"datetime": "2024-02-01T00:00:00Z"}, "assets": {}}

    def write(name, shift=0.):
        path = tmp_path / f"{name}.tif"
        data = np.full((64, 64), 5 if name == "scl" else 2000, dtype=np.uint16)
        if name == "scl":
            data[31, 31] = 9
        with rasterio.open(path, "w", driver="GTiff", width=64, height=64, count=1,
                           dtype="uint16", crs="EPSG:32719", transform=from_origin(left+shift, top, 20, 20),
                           nodata=0) as ds:
            ds.write(data, 1)
        scene["assets"][name] = {"href": str(path),
                                 "raster:bands": [{"scale": .0001, "offset": -.1, "nodata": 0}]}

    for name in names:
        write(name)
    raw, scl, qa, geometry = read_registered_scene(scene, -20., -69.)
    assert raw.shape == (6, 32, 32)
    assert qa["scientifically_accepted_proxy"]
    assert np.count_nonzero(scl == 9) == 1
    assert np.isnan(raw[:, scl == 9]).all()
    assert geometry["assets"]["scl"]["resampling"] == "nearest"
    assert geometry["assets"]["red"]["resampling"] == "bilinear"
    path = tmp_path / "sample.npy"
    np.save(path, raw)
    scl_path = path.with_suffix(".scl.npy")
    np.save(scl_path, scl)
    meta = {"acquisition_geometry": "registered_metric_grid_v2", "scl_sha256": file_sha256(scl_path),
            "scientific_patch_qa": qa}
    verify_registered_mask(path, meta)
    np.save(scl_path, np.ones_like(scl))
    with pytest.raises(ValueError, match="hash"):
        verify_registered_mask(path, meta)
    write("nir", shift=5.)
    with pytest.raises(ValueError, match="displaced"):
        read_registered_scene(scene, -20., -69.)


def test_registered_time_window_is_closed_and_ordered():
    assert str(validate_interval("2020-01-01/2021-01-01")[0]) == "2020-01-01"
    for invalid in ("2020-01-01/..", "2022-01-01/2021-01-01"):
        with pytest.raises(ValueError):
            validate_interval(invalid)
