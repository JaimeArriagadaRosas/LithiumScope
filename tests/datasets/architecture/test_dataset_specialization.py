"""Validate physical layout and compatibility of the dataset package."""
from importlib import import_module
from pathlib import Path
import pickle

import pytest

ROOT = Path(__file__).resolve().parents[3] / "src/lithiumscope/datasets"


def test_dataset_root_has_only_public_package_entry():
    assert {p.name for p in ROOT.glob("*.py")} == {"__init__.py"}
    assert all(len(list(p.glob("*.py"))) <= 20 for p in [ROOT, *ROOT.rglob("*")] if p.is_dir())


@pytest.mark.parametrize("old,new", [
    ("georoc_ingestion", "adapters.georoc.ingestion"),
    ("adapters.georoc_ingestion_transform", "adapters.georoc.transformation"),
    ("management.model_1_sources", "management.sources.model_1.source"),
    ("local_sources", "management.sources.local"),
    ("validation.analytical", "validation.chemistry.analytical"),
    ("validation.missing_values", "validation.tabular.missing_values"),
    ("validation.georoc_source_audit", "validation.georoc.source_audit"),
])
def test_legacy_aliases_are_identical(old, new):
    prefix = "lithiumscope.datasets."
    assert import_module(prefix + old) is import_module(prefix + new)


def test_legacy_pickled_manifest_class_is_resolvable():
    cls = pickle.loads(b"clithiumscope.datasets.manifest\nDatasetManifest\n.")
    from lithiumscope.datasets.provenance.manifest import DatasetManifest
    assert cls is DatasetManifest


def test_harmonization_is_public_facade():
    from lithiumscope.datasets.harmonization import merge_harmonized_sources
    from lithiumscope.datasets.harmonization.merging import merge_harmonized_sources as canonical
    assert merge_harmonized_sources is canonical
