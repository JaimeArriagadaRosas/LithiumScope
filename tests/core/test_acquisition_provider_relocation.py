from importlib import import_module
import pytest

@pytest.mark.parametrize("name", ["actions","diagnostics","download","download_links","export","html","log","models","payload","transfer"])
def test_georoc_query_alias(name):
    old="lithiumscope.tools.georoc.query.georoc_query_"+name
    new="lithiumscope.tools.acquisition.georoc.georoc_query_"+name
    assert import_module(old) is import_module(new)


def test_raster_alias():
    assert callable(import_module("lithiumscope.tools.acquisition.sentinel2.http_tiles").extract_http_patch)
    from lithiumscope.tools.acquisition.sentinel2.pilot import read_asset_isolated
    from lithiumscope.tools.acquisition.sentinel2.isolated import read_asset_isolated as canonical
    assert read_asset_isolated is canonical
