from importlib import import_module
import pytest

def test_historical_oof_recovery_import_is_read_only(tmp_path,monkeypatch):
    import pandas as pd
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(pd,"read_csv",lambda *a,**k:pytest.fail("Unexpected archive read"))
    old=import_module("lithiumscope.experiments.analysis.comparison.recover_oof_evidence")
    new=import_module("lithiumscope.experiments.model_2.historical.recover_oof_evidence")
    assert old is new
    assert not list(tmp_path.iterdir())
    with pytest.raises(ValueError,match="separate immutable"):
        new.recover(tmp_path,tmp_path)
