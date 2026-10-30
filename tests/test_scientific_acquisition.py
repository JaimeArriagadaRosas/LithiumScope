"""Bounded STAC retrieval and SCL policies."""
import numpy as np
import pytest
import subprocess
from lithiumscope.experiments.acquisition import (
    AssetTimeoutError, quality_from_scl, read_asset_isolated,
)


def test_scl_accepts_clear_land():
    result = quality_from_scl(np.full((8, 8), 5, dtype=np.uint8))
    assert result["quality_status"] == "ACCEPT"
    assert result["local_cloud_fraction"] == 0.0


def test_scl_rejects_cloud_shadow_and_nodata():
    scl = np.full((10, 10), 5, dtype=np.uint8)
    scl[:4, :] = 9
    result = quality_from_scl(scl)
    assert result["quality_status"] == "REJECT_CLOUD"
    assert result["local_cloud_fraction"] == pytest.approx(0.4)


def test_scl_requires_two_dimensions():
    with pytest.raises(ValueError):
        quality_from_scl(np.ones((2, 2, 2)))


def test_remote_asset_timeout_is_bounded(monkeypatch):
    def timeout(*args, **kwargs):
        assert kwargs["timeout"] == 7
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=7)
    monkeypatch.setattr(subprocess, "run", timeout)
    with pytest.raises(AssetTimeoutError):
        read_asset_isolated("https://example.com/red.tif", -23.1, -68.18, 640, 32,
                            timeout_seconds=7)
