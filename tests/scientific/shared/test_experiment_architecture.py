from importlib import import_module
from importlib.util import find_spec
from pathlib import Path

import pytest


@pytest.mark.parametrize("name", ["lithiumscope.model_1.methodology.dataset_comparison","lithiumscope.experiments.model_1.evaluation.dataset_diagnostics","lithiumscope.model_1.methodology.feot_expansion","lithiumscope.experiments.model_2.studies.grouped_training","lithiumscope.experiments.model_2.calibration.thresholds","lithiumscope.experiments.model_2.spatial.generalization","lithiumscope.experiments.model_2.calibration.nested","lithiumscope.experiments.integrated.compare_vnext_trained_runs","lithiumscope.experiments.integrated.verify_vnext_data_adequacy"])
def test_canonical_experiment_import(name):
    assert import_module(name).__name__ == name


def test_registry_has_only_canonical_owner():
    assert find_spec("lithiumscope.experiments.registry") is None
    assert import_module("lithiumscope.experiments.shared.registry").__name__ == "lithiumscope.experiments.shared.registry"


def test_experiment_domains_have_independent_packages():
    root = Path(__file__).resolve().parents[3] / "src/lithiumscope/experiments"
    assert all((root / name / "__init__.py").is_file()
               for name in ("model_1", "model_2", "integrated", "shared"))


def test_analysis_compatibility_package_removed():
    from lithiumscope.experiments import __path__
    assert all(not list((Path(root) / "analysis").rglob("*.py")) for root in __path__)


@pytest.mark.parametrize("group", ["model_1","model_2"])
def test_experimental_model_roots_are_domain_packages(group):
    root = Path(__file__).resolve().parents[3] / "src/lithiumscope/experiments" / group
    assert {p.name for p in root.glob("*.py")} == {"__init__.py"}
    assert (root.parent.parent / group / "eda" / "pipeline.py").is_file()


def test_source_lab_not_a_second_ordinary_eda():
    root = Path(__file__).resolve().parents[3] / "src/lithiumscope/experiments"
    assert not list((root / "georoc").rglob("*.py"))
    assert not list((root / "data_lab" / "eda").rglob("*.py"))
    assert (root / "data_lab/source_studies/georoc/candidate_eda.py").is_file()
    assert (root / "model_2/external_validation/imagery/batch.py").is_file()
