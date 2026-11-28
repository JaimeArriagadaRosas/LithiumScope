"""EDA modules have one canonical owner and do not fit estimators."""
from pathlib import Path
from importlib import import_module


def test_m1_eda_has_flat_responsibility_modules():
    package = import_module("lithiumscope.model_1.eda")
    root = Path(package.__file__).parent
    assert {p.name for p in root.glob("*.py")} == {
        "__init__.py", "pipeline.py", "statistics.py", "artifacts.py", "visualization.py", "quality_review.py", "spatial_review.py", "representativeness.py",
    }
    assert not any(list((root / name).rglob("*.py")) for name in ("reporting", "statistics", "visualization"))
    for module, name in (("statistics", "descriptive_tables"), ("artifacts", "write_descriptive_tables"), ("visualization", "render_profiles")):
        owner = import_module(f"lithiumscope.model_1.eda.{module}")
        assert getattr(owner, name).__module__ == owner.__name__
