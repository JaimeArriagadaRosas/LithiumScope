from __future__ import annotations

from pathlib import Path

from lithiumscope.core import config, paths


def test_explicit_home_override_keeps_installed_data_outside_site_packages(monkeypatch, tmp_path: Path) -> None:
    home = tmp_path / "lithiumscope-user"
    monkeypatch.setenv("LITHIUMSCOPE_HOME", str(home))
    assert paths._project_root() == home.resolve()


def test_packaged_defaults_copy_only_missing_files(monkeypatch, tmp_path: Path) -> None:
    package = tmp_path / "site-packages" / "lithiumscope"
    defaults = package / "_defaults"
    defaults.mkdir(parents=True)
    (defaults / "app.yaml").write_text("schema_version: 1\napp: {}\npaths: {}\n", encoding="utf-8")
    (defaults / "frozen.json").write_text('{"status":"new"}\n', encoding="utf-8")

    monkeypatch.setattr(config, "__file__", str(package / "core" / "configuration" / "loading.py"))
    local = tmp_path / "user-config"
    monkeypatch.setattr(config, "CONFIG_DIR", local)

    config._ensure_packaged_config_defaults()
    assert (local / "frozen.json").read_text(encoding="utf-8") == '{"status":"new"}\n'
    assert config.load_config("app")["schema_version"] == 1

    (local / "app.yaml").write_text("schema_version: 1\napp: {custom: true}\npaths: {}\n", encoding="utf-8")
    config._ensure_packaged_config_defaults()
    assert config.load_config("app")["app"]["custom"] is True
