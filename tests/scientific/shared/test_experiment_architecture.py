from importlib import import_module
from importlib.util import find_spec
from pathlib import Path

import pytest


@pytest.mark.parametrize("name", ["lithiumscope.model_1.methodology.dataset_comparison","lithiumscope.experiments.model_1.evaluation.dataset_diagnostics","lithiumscope.model_1.methodology.feot_expansion","lithiumscope.experiments.model_2.studies.grouped_training","lithiumscope.experiments.model_2.calibration.thresholds","lithiumscope.experiments.model_2.spatial.generalization","lithiumscope.experiments.model_2.calibration.nested","lithiumscope.experiments.integrated.compare_vnext_trained_runs","lithiumscope.experiments.integrated.verify_vnext_data_adequacy"])
def test_canonical_experiment_import(name):
    assert import_module(name).__name__ == name


def test_registry_has_only_canonical_owner():
    assert find_spec("lithiumscope." + "experiments.registry") is None
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


def test_retired_model_packages_have_no_source():
    root = Path(__file__).resolve().parents[3] / "src/lithiumscope/experiments"
    for name in ("model_1/targeted","model_1/balanced","model_1/augmentation","model_1/expansion",
                 "model_1/integral","model_2/historical","model_2/refinement_support"):
        assert not list((root/name).rglob("*.py"))


def test_source_studies_do_not_import_model_experiment_runners():
    import ast
    root = Path(__file__).resolve().parents[3] / "src/lithiumscope/experiments/data_lab/source_studies"
    forbidden = ("lithiumscope.experiments.model_1", "lithiumscope.experiments.model_2")
    for path in root.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_bytes())):
            names = [node.module or ""] if isinstance(node, ast.ImportFrom) else (
                [a.name for a in node.names] if isinstance(node, ast.Import) else [])
            assert not any(n.startswith(forbidden) for n in names), str(path)


def test_m1_shared_cohort_has_canonical_owner():
    from lithiumscope.experiments.model_1.protocols.cohorts import cohort
    assert cohort.__module__ == "lithiumscope.experiments.model_1.protocols.cohorts"
