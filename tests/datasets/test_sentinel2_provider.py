import numpy as np

from lithiumscope.model_2.data.sentinel2 import (
    Sentinel2Provider,
    SentinelConfig,
    SentinelProviderError,
    _masked_to_float,
)


class _Item:
    def __init__(self, payload):
        self.payload = payload

    def to_dict(self):
        return self.payload


class _Search:
    def __init__(self, items):
        self._items = items

    def items(self):
        return iter(self._items)


class _Client:
    def __init__(self):
        self.search_calls = 0

    def get_collection(self, collection):
        return {"id": collection}

    def search(self, **kwargs):
        self.search_calls += 1
        items = [
            _Item({"id": "cloudy", "properties": {"eo:cloud_cover": 20}, "assets": {}}),
            _Item({"id": "clear", "properties": {"eo:cloud_cover": 5}, "assets": {}}),
        ]
        return _Search(items)


def _config():
    return SentinelConfig(
        stac_url="https://example.test/v1",
        collection="sentinel-2-l2a",
        datetime="2020-01-01/..",
        cloud_cover_max=35,
        search_limit=20,
        request_timeout_seconds=5,
        max_retries=1,
        scene_cache_decimals=4,
        patch_size_m=640,
        patch_pixels=32,
        bands=("red", "nir"),
    )


def test_provider_validates_and_caches_coordinate_search():
    client = _Client()
    provider = Sentinel2Provider(_config(), client=client)
    provider.validate()
    first = provider.search_best_scene(-20.123456, -70.654321)
    second = provider.search_best_scene(-20.123459, -70.654319)

    assert first["id"] == "clear"
    assert second["id"] == "clear"
    assert client.search_calls == 1


def test_provider_validation_is_fail_fast():
    class BrokenClient:
        def get_collection(self, collection):
            raise RuntimeError("HTTP 400 invalid collection")

    provider = Sentinel2Provider(_config(), client=BrokenClient())

    try:
        provider.validate()
    except SentinelProviderError as exc:
        assert "HTTP 400" in str(exc)
    else:
        raise AssertionError("provider.validate() should fail")


def test_masked_uint16_can_be_filled_with_nan_after_float_conversion():
    data = np.ma.array(
        np.array([1, 2], dtype=np.uint16),
        mask=np.array([False, True]),
    )
    result = _masked_to_float(data)

    assert result.dtype == np.float32
    assert result[0] == 1.0
    assert np.isnan(result[1])
