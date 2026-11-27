from importlib import import_module
from importlib.util import find_spec
import pytest

@pytest.mark.parametrize("old,new",[["lithiumscope.experiments.validation.leakage_safe_m2","lithiumscope.experiments.model_2.spatial.leakage_control"],["lithiumscope.experiments.validation.seasonal2024_run","lithiumscope.experiments.model_2.temporal.seasonal_run"],["lithiumscope.experiments.validation.spatial_batch_run","lithiumscope.experiments.model_2.spatial.batch"],["lithiumscope.experiments.validation.spatial_control","lithiumscope.experiments.model_2.spatial.control"],["lithiumscope.experiments.validation.temporal_multiscene_run","lithiumscope.experiments.model_2.temporal.multiscene_run"]])
def test_historical_validation_import_is_read_only(old,new,monkeypatch,tmp_path):
    import pandas as pd
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(pd,"read_csv",lambda *a,**k:pytest.fail("import opened data"))
    assert find_spec(old) is None
    assert import_module(new).__name__ == new
    assert not list(tmp_path.iterdir())


def test_spatial_batch_import_does_not_require_optional_stac(monkeypatch):
    import builtins
    import sys

    canonical = "lithiumscope.experiments.model_2.spatial.batch"
    existing = sys.modules.get(canonical)
    monkeypatch.delitem(sys.modules, canonical, raising=False)
    original = builtins.__import__

    def without_stac(name, *args, **kwargs):
        if name == "pystac_client" or name.startswith("pystac_client."):
            raise ModuleNotFoundError("optional STAC client disabled for test")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_stac)
    try:
        module = import_module(canonical)
        assert callable(module.main)
    finally:
        if existing is not None:
            sys.modules[canonical] = existing
            parent = sys.modules[canonical.rsplit(".", 1)[0]]
            setattr(parent, "batch", existing)
        else:
            sys.modules.pop(canonical, None)
