from importlib import import_module
from importlib.util import find_spec
import pytest

@pytest.mark.parametrize("old,new",[
  [
    "lithiumscope.experiments.model_1.run_vnext_m1",
    "lithiumscope.model_1.methodology.dataset_comparison"
  ],
  [
    "lithiumscope.experiments.model_1.validate_feot_expansion_m1",
    "lithiumscope.model_1.methodology.feot_expansion"
  ]
])
def test_methodology_has_only_canonical_owner(old,new):
    assert find_spec(old) is None
    assert import_module(new).__name__ == new


def test_shared_quality_protocol_is_used_by_model_1():
    from lithiumscope.datasets.validation.chemistry.qc_protocols import protocol_masks
    from lithiumscope.model_1.methodology.qc_protocols import protocol_masks as m1_masks
    assert m1_masks is protocol_masks
