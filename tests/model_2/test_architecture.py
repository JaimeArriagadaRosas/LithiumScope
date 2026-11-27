"""Prevent reintroducing M2 architectural debt."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
M2 = ROOT / "src/lithiumscope/model_2"


def test_module_sizes_and_no_legacy_packages():
    files = list(M2.rglob("*.py"))
    assert files
    assert not [str(p) for p in files if len(p.read_text(encoding="utf-8").splitlines()) > 300]
    assert all(len(list(d.glob("*.py"))) <= 20 for d in {p.parent for p in files})
    for old in ("steps", "training/trainer.py", "training/prospectivity_model.py",
                "training/postprocess_prior_release.py", "training/postprocess_science.py",
                "methodology/ablations.py", "methodology/site_patch_lab.py"):
        target = M2 / old
        assert not target.is_file() and not list(target.rglob("*.py"))


def test_no_reverse_dependencies_or_module_aliases():
    for path in M2.rglob("*.py"):
        relative = path.relative_to(M2).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
            elif isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                assert not (node.value.id == "sys" and node.attr == "modules"), relative
        assert not any(".experiments." in name for name in imports), relative
        if relative.startswith("training/execution/"):
            assert not any(name.endswith("model_2.training.competition") for name in imports), relative
        if relative.startswith(("evaluation/", "diagnostics/")):
            assert not any(name.endswith("model_2.training.postprocess") for name in imports), relative
