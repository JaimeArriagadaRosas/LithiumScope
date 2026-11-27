"""Canonical acquisition and adaptation preserve the existing public imports."""
from importlib import import_module

import pytest


@pytest.mark.parametrize("name", [
    "tools.acquisition.tabular", "tools.acquisition.georoc.query_contract",
    "tools.acquisition.georoc.query_flow", "tools.acquisition.georoc.filtered",
    "datasets.adapters.georoc.material", "datasets.adapters.georoc.schema",
    "datasets.adapters.georoc.transformation", "datasets.adapters.georoc.ingestion",
])
def test_canonical_acquisition_module_import(name):
    assert import_module(f"lithiumscope.{name}")


def test_checksum_reuses_shared_streaming_implementation(tmp_path):
    from lithiumscope.core.hashing import file_sha256
    path = tmp_path / "test.csv"
    path.write_bytes(b"Li_icpms\n12\n")
    assert file_sha256(path, chunk_size=1) == file_sha256(path)


def test_georoc_adapters_are_fingerprinted():
    from lithiumscope.core.execution.resume import _scientific_code_files
    files = {p.as_posix() for p in _scientific_code_files("model_1")}
    assert any(p.endswith("datasets/adapters/georoc/transformation.py") for p in files)
