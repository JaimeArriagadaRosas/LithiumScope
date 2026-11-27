from importlib import import_module
import pytest

def test_historical_oof_recovery_import_is_read_only(tmp_path,monkeypatch):
    import pandas as pd
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(pd,"read_csv",lambda *a,**k:pytest.fail("Unexpected archive read"))
    new=import_module("lithiumscope.experiments.model_2.evaluation.oof_recovery")
    assert new.__name__ == "lithiumscope.experiments.model_2.evaluation.oof_recovery"
    assert not list(tmp_path.iterdir())
    with pytest.raises(ValueError,match="separate immutable"):
        new.recover(tmp_path,tmp_path)


def test_scope_closure_import_is_read_only(tmp_path,monkeypatch):
    import pandas as pd
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(pd,"read_csv",lambda *a,**k:pytest.fail("Unexpected simulation read"))
    from importlib.util import find_spec
    assert find_spec("lithiumscope.experiments.reporting.final_scope_closure") is None
    new=import_module("lithiumscope.experiments.integrated.final_scope_closure")
    assert new.__name__ == "lithiumscope.experiments.integrated.final_scope_closure"
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("new", ["lithiumscope.experiments.model_2.studies.grouped_training","lithiumscope.experiments.model_2.evaluation.learning_curves"])
def test_historical_protocols_resolve_without_eager_facade_cycle(new):
    assert import_module(new).__name__ == new
    import ast
    from pathlib import Path
    path=Path(import_module(new).__file__)
    forbidden={"lithiumscope.experiments.model_2.studies.grouped_training","lithiumscope.experiments.integrated.verify_vnext_data_adequacy"}
    retired_prefix="lithiumscope.experiments." + "analysis."
    assert not any(isinstance(node,ast.ImportFrom) and (node.module in forbidden or (node.module or "").startswith(retired_prefix)) for node in ast.parse(path.read_bytes()).body)
