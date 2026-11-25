"""Ensure canonical diagnostic moves preserve import identity and API."""
from importlib import import_module

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
  [
    "lithiumscope.model_1.evaluation.domain",
    "lithiumscope.model_1.diagnostics.domain"
  ],
  [
    "lithiumscope.model_1.evaluation.group_intervals",
    "lithiumscope.model_1.diagnostics.group_intervals"
  ],
  [
    "lithiumscope.model_1.evaluation.high_lithium",
    "lithiumscope.model_1.diagnostics.high_lithium"
  ],
  [
    "lithiumscope.model_1.evaluation.robustness",
    "lithiumscope.model_1.diagnostics.robustness"
  ],
  [
    "lithiumscope.model_1.methodology.dual_release_metrics",
    "lithiumscope.model_1.diagnostics.dual_release_metrics"
  ],
  [
    "lithiumscope.model_2.evaluation.generalization",
    "lithiumscope.model_2.diagnostics.generalization"
  ],
  [
    "lithiumscope.model_2.evaluation.independence",
    "lithiumscope.model_2.diagnostics.independence"
  ],
  [
    "lithiumscope.model_2.evaluation.paired_uncertainty",
    "lithiumscope.model_2.diagnostics.paired_uncertainty"
  ],
  [
    "lithiumscope.model_2.evaluation.spatial_distance_sensitivity",
    "lithiumscope.model_2.diagnostics.spatial_distance_sensitivity"
  ],
  [
    "lithiumscope.model_2.methodology.patch_missingness",
    "lithiumscope.model_2.diagnostics.patch_missingness"
  ],
  [
    "lithiumscope.model_2.methodology.spectral.spectral_overfit_inputs",
    "lithiumscope.model_2.diagnostics.spectral_overfit_inputs"
  ],
  [
    "lithiumscope.model_2.methodology.spectral_overfit_pilot",
    "lithiumscope.model_2.diagnostics.spectral_overfit"
  ],
  [
    "lithiumscope.model_2.methodology.validation.domain",
    "lithiumscope.model_2.diagnostics.domain"
  ],
  [
    "lithiumscope.model_2.training.diagnostic_audit",
    "lithiumscope.model_2.diagnostics.competition_audit"
  ],
  [
    "lithiumscope.model_2.training.diagnostic_inputs",
    "lithiumscope.model_2.diagnostics.competition_inputs"
  ],
  [
    "lithiumscope.model_2.training.diagnostic_metrics",
    "lithiumscope.model_2.diagnostics.competition_metrics"
  ],
  [
    "lithiumscope.model_2.training.diagnostic_release_comparison",
    "lithiumscope.model_2.diagnostics.release_comparison"
  ]
])
def test_relocated_module_identity(old,new):
    assert import_module(old) is import_module(new)


def test_reproducibility_package_api():
    from lithiumscope.core.reproducibility import canonical_json_hash, runtime_fingerprint, set_global_seed
    from lithiumscope.core import hashing
    assert canonical_json_hash({"b":2,"a":1})==canonical_json_hash({"a":1,"b":2})
    assert callable(runtime_fingerprint) and callable(set_global_seed)
    assert callable(hashing.file_sha256)
