from types import SimpleNamespace

import lithiumscope.runtime.dependencies as dependencies


def _fake_find_spec_factory(
    missing: set[str],
):
    def find_spec(name: str):
        return None if name in missing else object()
    return find_spec


def test_dependency_repair_installs_only_base_for_core_gap(
    monkeypatch,
    tmp_path,
):
    missing = {"reportlab"}
    monkeypatch.setattr(
        dependencies,
        "PROJECT_ROOT",
        tmp_path,
    )
    monkeypatch.setattr(
        dependencies,
        "_virtualenv_active",
        lambda: True,
    )
    monkeypatch.setattr(
        dependencies.importlib.util,
        "find_spec",
        _fake_find_spec_factory(missing),
    )

    commands = []

    def run(command, **kwargs):
        commands.append(command)
        missing.clear()
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(
        dependencies.subprocess,
        "run",
        run,
    )
    monkeypatch.setattr(
        dependencies,
        "_LAST_REPAIR_RESULT",
        None,
    )

    result = dependencies.ensure_runtime_dependencies(
        verbose=False
    )

    assert result.success is True
    assert result.extras == ()
    assert commands
    assert commands[0][-1] == "."
    assert commands[0][-2] == "-e"


def test_dependency_repair_adds_only_missing_extras(
    monkeypatch,
    tmp_path,
):
    missing = {
        "xgboost",
        "rasterio",
    }
    monkeypatch.setattr(
        dependencies,
        "PROJECT_ROOT",
        tmp_path,
    )
    monkeypatch.setattr(
        dependencies,
        "_virtualenv_active",
        lambda: True,
    )
    monkeypatch.setattr(
        dependencies.importlib.util,
        "find_spec",
        _fake_find_spec_factory(missing),
    )

    commands = []

    def run(command, **kwargs):
        commands.append(command)
        missing.clear()
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(
        dependencies.subprocess,
        "run",
        run,
    )
    monkeypatch.setattr(
        dependencies,
        "_LAST_REPAIR_RESULT",
        None,
    )

    result = dependencies.ensure_runtime_dependencies(
        verbose=False
    )

    assert result.success is True
    assert result.extras == (
        "ml",
        "imagery",
    )
    assert commands[0][-1] == ".[ml,imagery]"


def test_dependency_repair_does_not_modify_global_python(
    monkeypatch,
):
    missing = {"reportlab"}
    monkeypatch.setattr(
        dependencies.importlib.util,
        "find_spec",
        _fake_find_spec_factory(missing),
    )
    monkeypatch.setattr(
        dependencies,
        "_virtualenv_active",
        lambda: False,
    )
    monkeypatch.setattr(
        dependencies,
        "_LAST_REPAIR_RESULT",
        None,
    )

    result = dependencies.ensure_runtime_dependencies(
        verbose=False
    )

    assert result.attempted is False
    assert result.success is False
    assert result.skipped_reason == (
        "virtualenv_not_active"
    )
