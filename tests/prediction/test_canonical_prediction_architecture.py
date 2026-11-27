"""Prediction has canonical owners rather than compatibility modules."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "src/lithiumscope"


def test_prediction_and_reporting_boundaries():
    prediction = ROOT / "prediction"
    reporting = ROOT / "reporting"
    assert {p.name for p in prediction.glob("*.py")} == {"__init__.py"}
    assert not list((prediction / "exports").rglob("*.py"))
    assert not list((prediction / "candidates").rglob("*.py"))
    for path in [*prediction.rglob("*.py"), *reporting.rglob("*.py")]:
        source = path.read_text(encoding="utf-8")
        assert len(source.splitlines()) <= 300, str(path)
        assert len(list(path.parent.glob("*.py"))) <= 20
        tree = ast.parse(source)
        assert not any(isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
                       and n.value.id == "sys" and n.attr == "modules" for n in ast.walk(tree)), str(path)
        if path.is_relative_to(reporting):
            for n in ast.walk(tree):
                names = [n.module or ""] if isinstance(n, ast.ImportFrom) else (
                    [a.name for a in n.names] if isinstance(n, ast.Import) else [])
                assert not any(name.startswith(("lithiumscope.prediction", "lithiumscope.model_1",
                                               "lithiumscope.model_2", "lithiumscope.experiments"))
                               for name in names), str(path)


def test_prediction_subpackages_are_present():
    assert all((ROOT / "prediction" / name / "__init__.py").is_file()
               for name in ("inputs", "execution", "analysis", "demonstration"))


def test_demo_publication_implementation_is_in_distribution():
    from lithiumscope.distribution.demonstration_bundle import build_demonstration_release_bundle
    assert build_demonstration_release_bundle.__module__ == "lithiumscope.distribution.demonstration_bundle"
