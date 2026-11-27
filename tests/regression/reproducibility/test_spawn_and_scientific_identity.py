"""Structural regression contracts for spawn and relocated scientific fingerprints."""
import ast
from pathlib import Path
from lithiumscope.core.execution.resume import _scientific_code_files

def test_spawn_entrypoint_does_not_bootstrap_dependencies_on_import():
    root=next(parent for parent in Path(__file__).resolve().parents if (parent / "pyproject.toml").is_file())
    tree=ast.parse((root/"main.py").read_bytes())
    assert not any(isinstance(node,ast.ImportFrom) and node.module=="lithiumscope.runtime.dependencies" for node in tree.body)
    assert any(isinstance(node,ast.FunctionDef) and node.name=="launch" for node in tree.body)

def test_relocated_scientific_contracts_remain_in_fingerprint():
    for model in ("model_1","model_2"):
        files={p.as_posix() for p in _scientific_code_files(model)}
        for suffix in ("core/reproducibility/__init__.py","core/reproducibility/hashing.py",
                       "core/scientific/regression_metrics.py","core/scientific/classification_metrics.py"):
            assert any(p.endswith(suffix) for p in files)
        assert any(p.endswith(f"{model}/training/competition.py") for p in files)
        assert any(p.endswith(f"{model}/evaluation/selection.py") for p in files)
