from importlib import import_module
from pathlib import Path

import pytest


@pytest.mark.parametrize("name", ["lithiumscope.model_1.methodology.dataset_comparison","lithiumscope.experiments.model_1.evaluation.dataset_diagnostics","lithiumscope.model_1.methodology.feot_expansion","lithiumscope.experiments.model_2.run_vnext_m2","lithiumscope.experiments.model_2.run_vnext_m2_threshold","lithiumscope.experiments.model_2.m2_generalization_pilot","lithiumscope.experiments.model_2.m2_nested_calibration","lithiumscope.experiments.integrated.compare_vnext_trained_runs","lithiumscope.experiments.integrated.verify_vnext_data_adequacy"])
def test_canonical_experiment_import(name):
    assert import_module(name).__name__ == name


def test_registry_alias():
    assert import_module("lithiumscope.experiments.registry") is import_module("lithiumscope.experiments.shared.registry")


def test_experiment_domains_have_independent_packages():
    root = Path(__file__).resolve().parents[3] / "src/lithiumscope/experiments"
    assert all((root / name / "__init__.py").is_file()
               for name in ("model_1", "model_2", "integrated", "shared"))


def test_analysis_compatibility_package_removed():
    from lithiumscope.experiments import __path__
    assert all(not list((Path(root) / "analysis").rglob("*.py")) for root in __path__)
