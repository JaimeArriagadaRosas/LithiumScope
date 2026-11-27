"""Ensure canonical diagnostic moves preserve import identity and API."""
from importlib import import_module
from importlib.util import find_spec

import pytest


@pytest.mark.parametrize("old,new", [
  [
    "lithiumscope.core.hashing",
    "lithiumscope.core.reproducibility.hashing"
  ],
  [
    "lithiumscope.core.presentation.visualization",
    "lithiumscope.core.visualization"
  ],
])
def test_relocated_module_identity(old,new):
    assert import_module(old) is import_module(new)


def test_reproducibility_package_api():
    from lithiumscope.core.reproducibility import canonical_json_hash, runtime_fingerprint, set_global_seed
    from lithiumscope.core import hashing
    assert canonical_json_hash({"b":2,"a":1})==canonical_json_hash({"a":1,"b":2})
    assert callable(runtime_fingerprint) and callable(set_global_seed)
    assert callable(hashing.file_sha256)


@pytest.mark.parametrize("old", ["model_1.evaluation.domain","model_1.evaluation.group_intervals","model_1.evaluation.high_lithium","model_1.evaluation.robustness","model_1.methodology.dual_release_metrics"])
def test_m1_diagnostic_compatibility_modules_retired(old):
    assert find_spec("lithiumscope." + old) is None


@pytest.mark.parametrize("old", ['model_2.evaluation.generalization', 'model_2.evaluation.independence', 'model_2.evaluation.metrics', 'model_2.evaluation.paired_uncertainty', 'model_2.evaluation.spatial_distance_sensitivity', 'model_2.evaluation.reports', 'model_2.training.diagnostic_audit', 'model_2.training.diagnostic_inputs', 'model_2.training.diagnostic_metrics', 'model_2.training.diagnostic_release_comparison', 'model_2.methodology.patch_missingness', 'model_2.methodology.spectral.spectral_overfit_inputs', 'model_2.methodology.spectral_overfit_pilot', 'model_2.methodology.validation.domain'])
def test_m2_diagnostic_compatibility_modules_retired(old):
    assert find_spec("lithiumscope." + old) is None
