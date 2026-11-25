from importlib import import_module
from pathlib import Path

import pytest


@pytest.mark.parametrize(("old", "new"), [
    ("lithiumscope.experiments.analysis.run_vnext_m1", "lithiumscope.experiments.model_1.run_vnext_m1"),
    ("lithiumscope.experiments.analysis.run_vnext_m1_diagnostics", "lithiumscope.experiments.model_1.run_vnext_m1_diagnostics"),
    ("lithiumscope.experiments.analysis.validate_feot_expansion_m1", "lithiumscope.experiments.model_1.validate_feot_expansion_m1"),
    ("lithiumscope.experiments.analysis.run_vnext_m2", "lithiumscope.experiments.model_2.run_vnext_m2"),
    ("lithiumscope.experiments.analysis.run_vnext_m2_threshold", "lithiumscope.experiments.model_2.run_vnext_m2_threshold"),
    ("lithiumscope.experiments.analysis.m2_generalization_pilot", "lithiumscope.experiments.model_2.m2_generalization_pilot"),
    ("lithiumscope.experiments.analysis.m2_nested_calibration", "lithiumscope.experiments.model_2.m2_nested_calibration"),
    ("lithiumscope.experiments.analysis.compare_vnext_trained_runs", "lithiumscope.experiments.integrated.compare_vnext_trained_runs"),
    ("lithiumscope.experiments.analysis.verify_vnext_data_adequacy", "lithiumscope.experiments.integrated.verify_vnext_data_adequacy"),
    ("lithiumscope.experiments.registry", "lithiumscope.experiments.shared.registry"),
])
def test_legacy_experiment_import_resolves_canonical_module(old, new):
    assert import_module(old) is import_module(new)


def test_experiment_domains_have_independent_packages():
    root = Path(__file__).resolve().parents[3] / "src/lithiumscope/experiments"
    assert all((root / name / "__init__.py").is_file()
               for name in ("model_1", "model_2", "integrated", "shared"))
