import ast
from importlib import import_module
from pathlib import Path

import pandas as pd
import pytest

from lithiumscope.datasets.management.sources.model_2 import require_frozen_manifest
from lithiumscope.datasets.provenance.manifest import build_tabular_manifest


@pytest.mark.parametrize("name", [
    "datasets.management.registry", "datasets.management.status",
    "datasets.management.sources.local", "datasets.management.sources.model_1",
    "runtime.dataset_provisioning", "datasets.provenance.manifest", "datasets.validation.files",
])
def test_canonical_management_imports(name):
    assert import_module(f"lithiumscope.{name}")


def test_datasets_have_no_direct_model_imports():
    root = Path(__file__).resolve().parents[3] / "src/lithiumscope/datasets"
    for path in root.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8-sig"))):
            modules = [node.module or ""] if isinstance(node, ast.ImportFrom) else (
                [a.name for a in node.names] if isinstance(node, ast.Import) else [])
            assert not any(m.startswith(("lithiumscope.model_1", "lithiumscope.model_2")) for m in modules), path


def test_manifest_reports_missing_and_binds_ordered_patch_bytes(tmp_path):
    a = tmp_path / "a.npy"
    b = tmp_path / "b.npy"
    a.write_bytes(b"a")
    b.write_bytes(b"b")
    source = tmp_path / "manifest.csv"
    frame = pd.DataFrame({"image_path": [str(a), str(b)]})
    frame.to_csv(source, index=False)
    def build():
        return build_tabular_manifest(source, "m2", "model_2", "test")
    first = build()
    assert first.state == "ready"
    frame.iloc[::-1].to_csv(source, index=False)
    assert first.metadata["linked_files_sha256"] != build().metadata["linked_files_sha256"]
    b.unlink()
    partial = build()
    assert partial.state == "partial"
    assert partial.metadata["linked_file_missing_count"] == 1


def test_frozen_cohort_is_read_only_and_rejects_invalid_identity(tmp_path):
    source = tmp_path / "m2.csv"
    frame = pd.DataFrame({"Li_icpms": [10., 20.], "spatial_group": ["a", "b"],
                          "sample_id": ["s1", "s2"], "image_path": ["a.npy", "b.npy"]})
    frame.to_csv(source, index=False)
    cfg = {"manifest_path": "m2.csv", "frozen_expected_rows": 2}
    before = source.read_bytes()
    assert require_frozen_manifest(cfg, tmp_path) == source
    assert source.read_bytes() == before
    frame.loc[1, "sample_id"] = "s1"
    frame.to_csv(source, index=False)
    with pytest.raises(ValueError, match="duplicate"):
        require_frozen_manifest(cfg, tmp_path)
    with pytest.raises(ValueError, match="outside"):
        require_frozen_manifest({**cfg, "manifest_path": "../old.csv"}, tmp_path)


def test_source_management_has_no_nested_model_packages_or_runtime_import():
    root = Path(__file__).resolve().parents[3] / "src/lithiumscope/datasets/management"
    assert not list((root / "sources/model_1").rglob("*.py"))
    assert not list((root / "sources/model_2").rglob("*.py"))
    assert not (root / "paths.py").exists()
    tree = ast.parse((root / "sources/diagnostic.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        imports = [node.module or ""] if isinstance(node, ast.ImportFrom) else (
            [alias.name for alias in node.names] if isinstance(node, ast.Import) else [])
        assert not any(name.startswith("lithiumscope.runtime") for name in imports)
