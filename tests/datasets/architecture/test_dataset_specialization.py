"""Dataset architecture: canonical modules, portable paths and no compatibility shims."""
from pathlib import Path
from importlib import import_module, util
import pickle

import pytest

ROOT = Path(__file__).resolve().parents[3] / "src/lithiumscope/datasets"


def test_dataset_root_has_only_public_package_entry():
    assert {p.name for p in ROOT.glob("*.py")} == {"__init__.py"}
    assert not list(ROOT.rglob("_legacy"))
    assert all(len(list(p.glob("*.py"))) <= 20 for p in [ROOT, *ROOT.rglob("*")] if p.is_dir())


@pytest.mark.parametrize("name", [
    "adapters.georoc.ingestion", "adapters.georoc.transformation",
    "management.sources.model_1.source", "management.sources.local",
    "validation.chemistry.analytical", "validation.tabular.missing_values",
    "validation.georoc.source_audit",
])
def test_canonical_modules_import(name):
    assert import_module("lithiumscope.datasets." + name)


@pytest.mark.parametrize("name", ["georoc_ingestion", "manifest", "local_sources",
    "adapters.georoc_ingestion", "management.model_1_sources", "validation.analytical"])
def test_obsolete_import_paths_are_retired(name):
    assert util.find_spec("lithiumscope.datasets." + name) is None


def test_canonical_manifest_class_roundtrip():
    from lithiumscope.datasets.provenance.manifest import DatasetManifest
    assert pickle.loads(pickle.dumps(DatasetManifest)) is DatasetManifest


def test_harmonization_is_public_facade():
    from lithiumscope.datasets.harmonization import merge_harmonized_sources
    from lithiumscope.datasets.harmonization.merging import merge_harmonized_sources as canonical
    assert merge_harmonized_sources is canonical


def test_relative_source_paths_are_portable(tmp_path, monkeypatch):
    from lithiumscope.datasets.management.sources.model_1.configuration import project_path
    first, second = tmp_path / "first clone", tmp_path / "second clone"
    first.mkdir()
    second.mkdir()
    monkeypatch.chdir(first)
    relative = "data/processed/model_1/training.csv"
    assert project_path(relative, first) == first / relative
    assert project_path(relative, second) == second / relative
