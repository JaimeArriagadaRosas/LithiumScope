from importlib import import_module
import pytest

@pytest.mark.parametrize("old,new",[["lithiumscope.experiments.validation.leakage_safe_m2","lithiumscope.experiments.model_2.leakage_safe_m2"],["lithiumscope.experiments.validation.seasonal2024_run","lithiumscope.experiments.model_2.historical.seasonal2024_run"],["lithiumscope.experiments.validation.spatial_batch_run","lithiumscope.experiments.model_2.historical.spatial_batch_run"],["lithiumscope.experiments.validation.spatial_control","lithiumscope.experiments.model_2.historical.spatial_control"],["lithiumscope.experiments.validation.temporal_multiscene_run","lithiumscope.experiments.model_2.historical.temporal_multiscene_run"]])
def test_historical_validation_import_is_read_only(old,new,monkeypatch,tmp_path):
    import pandas as pd
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(pd,"read_csv",lambda *a,**k:pytest.fail("import opened data"))
    assert import_module(old) is import_module(new)
    assert not list(tmp_path.iterdir())
