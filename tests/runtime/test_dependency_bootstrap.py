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



def test_dependency_repair_captures_installer_output(
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

    def run(command, **kwargs):
        assert kwargs["stdout"] is dependencies.subprocess.PIPE
        assert kwargs["stderr"] is dependencies.subprocess.STDOUT
        assert kwargs["text"] is True
        missing.clear()
        return SimpleNamespace(
            returncode=0,
            stdout="pip detail hidden from normal output",
        )

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
    assert (
        result.installer_output_tail[-1]
        == "pip detail hidden from normal output"
    )

def test_core_bootstrap_does_not_install_optional_packages(monkeypatch):
    monkeypatch.setattr(dependencies.importlib.util, "find_spec",
                        _fake_find_spec_factory({"torch", "pytorch_tabnet", "pystac_client"}))
    monkeypatch.setattr(dependencies.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(AssertionError("No installation expected")))
    result = dependencies.ensure_runtime_dependencies(verbose=False, include_optional=False)
    assert result.success and not result.attempted


def test_training_environment_is_model_specific(monkeypatch):
    from lithiumscope.runtime import preboot
    monkeypatch.setattr(dependencies.importlib.util, "find_spec",
                        _fake_find_spec_factory({"torch", "pytorch_tabnet"}))
    assert preboot.missing_training_dependencies("model_2") == []
    assert set(preboot.missing_training_dependencies("model_1")) == {"PyTorch", "PyTorch TabNet"}


def test_preboot_preserves_placeholders(tmp_path, monkeypatch):
    from lithiumscope.runtime import preboot
    p = tmp_path / ".gitkeep"
    p.touch()
    monkeypatch.setattr(preboot, "DATA_DIR", tmp_path)
    assert preboot.cleanup_gitkeep_placeholders() == 0
    assert p.exists()
