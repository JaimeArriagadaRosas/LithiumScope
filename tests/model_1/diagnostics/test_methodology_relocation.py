from importlib import import_module
import pytest

@pytest.mark.parametrize("old,new",[
  [
    "lithiumscope.experiments.model_1.run_vnext_m1",
    "lithiumscope.model_1.methodology.dataset_comparison"
  ],
  [
    "lithiumscope.experiments.model_1.validate_feot_expansion_m1",
    "lithiumscope.model_1.methodology.feot_expansion"
  ],
  [
    "lithiumscope.experiments.analysis.protocols.georoc_qc_protocols",
    "lithiumscope.model_1.methodology.qc_protocols"
  ]
])
def test_methodology_alias_identity(old,new):
    assert import_module(old) is import_module(new)
