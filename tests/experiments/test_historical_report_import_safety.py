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


def test_scope_closure_import_is_read_only(tmp_path,monkeypatch):
    import pandas as pd
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(pd,"read_csv",lambda *a,**k:pytest.fail("Unexpected simulation read"))
    old=import_module("lithiumscope.experiments.reporting.final_scope_closure")
    new=import_module("lithiumscope.experiments.integrated.final_scope_closure")
    assert old is new
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("old,new",[["lithiumscope.experiments.analysis.protocols.run_vnext_m2_run_experiments","lithiumscope.experiments.model_2.historical.vnext_experiments"],["lithiumscope.experiments.analysis.protocols.verify_vnext_data_adequacy_run_m2_learning_curve","lithiumscope.experiments.model_2.historical.adequacy_learning_curve"]])
def test_historical_protocols_resolve_without_eager_facade_cycle(old,new):
    assert import_module(old) is import_module(new)
    import ast
    from pathlib import Path
    path=Path(import_module(new).__file__)
    forbidden={"lithiumscope.experiments.analysis.run_vnext_m2","lithiumscope.experiments.analysis.verify_vnext_data_adequacy","lithiumscope.experiments.model_2.run_vnext_m2","lithiumscope.experiments.integrated.verify_vnext_data_adequacy"}
    assert not any(isinstance(node,ast.ImportFrom) and node.module in forbidden for node in ast.parse(path.read_bytes()).body)
