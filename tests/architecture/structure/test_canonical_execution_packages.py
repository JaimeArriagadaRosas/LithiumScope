"""Verify one canonical implementation and stable historical import identity."""
from importlib import import_module
from pathlib import Path

import pytest


@pytest.mark.parametrize(("old", "new"), [
    ("core.config", "core.configuration.loading"),
    ("core.config_schema", "core.configuration.schema"),
    ("core.checkpoints", "core.execution.checkpoints"),
    ("core.run_context", "core.execution.context"),
    ("core.run_resume", "core.execution.resume"),
    ("core.experiment_tracker", "core.execution.tracker"),
    ("core.code_size_audit", "tools.inspection.code_size"),
])
def test_canonical_module_identity(old, new):
    assert import_module(f"lithiumscope.{old}") is import_module(f"lithiumscope.{new}")


def test_packaged_config_resolution_uses_package_root():
    from lithiumscope.core.configuration import loading
    assert Path(loading.__file__).resolve().parents[2].name == "lithiumscope"


def test_scientific_fingerprint_includes_canonical_checkpoints_and_configuration():
    from lithiumscope.core.execution.resume import _scientific_code_files
    names = {p.as_posix() for p in _scientific_code_files("model_1")}
    for suffix in ("core/execution/checkpoints.py", "core/configuration/loading.py",
                   "core/configuration/schema.py"):
        assert any(p.endswith(suffix) for p in names), suffix
