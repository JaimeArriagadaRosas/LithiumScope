"""Canonical acquisition and adaptation preserve the existing public imports."""
from importlib import import_module

import pytest


@pytest.mark.parametrize(("old", "new"), [
    ("datasets.downloader", "tools.acquisition.tabular"),
    ("datasets.georoc_query_contract", "tools.acquisition.georoc.query_contract"),
    ("datasets.georoc_query_flow", "tools.acquisition.georoc.query_flow"),
    ("datasets.georoc_filtered_acquisition", "tools.acquisition.georoc.filtered"),
    ("tools.georoc.processing.georoc_material_parser", "datasets.adapters.georoc_material"),
    ("datasets.georoc_ingestion_schema", "datasets.adapters.georoc_ingestion_schema"),
    ("datasets.georoc_ingestion_transform", "datasets.adapters.georoc_ingestion_transform"),
    ("datasets.georoc_ingestion", "datasets.adapters.georoc_ingestion"),
])
def test_canonical_acquisition_module_identity(old, new):
    assert import_module(f"lithiumscope.{old}") is import_module(f"lithiumscope.{new}")


def test_checksum_reuses_shared_streaming_implementation(tmp_path):
    from lithiumscope.datasets.checksum import sha256
    from lithiumscope.core.hashing import file_sha256
    path = tmp_path / "test.csv"
    path.write_bytes(b"Li_icpms\n12\n")
    assert sha256 is file_sha256
    assert sha256(path, chunk_size=1) == file_sha256(path)


def test_georoc_adapters_are_fingerprinted():
    from lithiumscope.core.execution.resume import _scientific_code_files
    files = {p.as_posix() for p in _scientific_code_files("model_1")}
    assert any(p.endswith("datasets/adapters/georoc_ingestion_transform.py") for p in files)
