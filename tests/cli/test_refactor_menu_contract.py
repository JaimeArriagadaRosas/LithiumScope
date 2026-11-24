from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from lithiumscope.cli import menu, predict_command, train_command, model_release_command
from lithiumscope.runtime import lifecycle


def test_main_menu_approved_order_and_dispatch(monkeypatch, capsys):
    choices = iter(["1", "2", "0"])
    calls = []
    monkeypatch.setattr(menu, "choose", lambda *_: next(choices))
    monkeypatch.setattr(menu, "clear_screen", lambda: None)
    monkeypatch.setattr(menu, "ensure_runtime_directories", lambda: None)
    monkeypatch.setattr(menu, "configure_logging", lambda: None)
    monkeypatch.setattr(menu.predict_command, "run", lambda: calls.append("prediction") or False)
    monkeypatch.setattr(menu.train_command, "run", lambda: calls.append("training") or False)
    assert menu.main() == 0
    assert calls == ["prediction", "training"]
    out = capsys.readouterr().out
    assert "1. Realizar predicción" in out
    assert "2. Entrenar modelos" in out
    assert "5. Laboratorio de diagnóstico" in out


@pytest.mark.parametrize("option,method", [("1", "_run_complete"), ("2", "_run_demonstration"),
                                         ("3", "_run_model_1"), ("4", "_run_model_2")])
def test_prediction_submenu_dispatch(monkeypatch, option, method):
    calls = []
    monkeypatch.setattr(predict_command, "choose", lambda *_: option)
    for name in ("_run_complete", "_run_demonstration", "_run_model_1", "_run_model_2"):
        monkeypatch.setattr(predict_command, name, lambda n=name: calls.append(n))
    predict_command.run()
    assert calls == [method]


def test_preboot_pause_precedes_menu(monkeypatch):
    calls = []
    monkeypatch.setattr(lifecycle, "get_shutdown_manager", lambda: SimpleNamespace(
        install=lambda: None, finalize=lambda: None))
    monkeypatch.setattr(lifecycle, "install_warning_capture", lambda: None)
    monkeypatch.setattr(lifecycle, "configure_headless_matplotlib", lambda: "Agg")
    monkeypatch.setattr(lifecycle, "recover_abandoned_runs", lambda: [])
    monkeypatch.setattr(lifecycle, "run_preboot",
                        lambda **_: calls.append("preboot") or SimpleNamespace(core_ready=True))
    monkeypatch.setattr(lifecycle, "_print_session_summary", lambda: None)
    monkeypatch.setattr("builtins.input", lambda _: calls.append("enter") or "")
    assert lifecycle.run_application(lambda: calls.append("menu") or 0) == 0
    assert calls == ["preboot", "enter", "menu"]


def test_cancelled_training_removes_only_new_session(monkeypatch, tmp_path, capsys):
    old = tmp_path / "old-session"
    old.mkdir()
    (old / "evidence").write_text("keep")
    new = tmp_path / "new-session"
    @contextmanager
    def fake_session(_):
        new.mkdir()
        (new / "status.json").write_text("{}")
        yield new
    monkeypatch.setattr(train_command, "training_session", fake_session)
    monkeypatch.setattr(train_command, "_run_impl", lambda: False)
    prompts = []
    monkeypatch.setattr("builtins.input", lambda prompt: prompts.append(prompt) or "")
    assert train_command.run() is False
    assert not new.exists()
    assert (old / "evidence").read_text() == "keep"
    assert prompts == ["Se ha limpiado el registro de entrenamiento. Presione ENTER para continuar"]


def test_release_confirmation_cancellation_does_not_download(monkeypatch, tmp_path):
    release = SimpleNamespace(tag_name="test", published_at="2026-01-01", asset_size=42,
                              asset_name="test-artifacts.zip")
    monkeypatch.setattr(model_release_command, "list_model_releases", lambda: [release])
    monkeypatch.setattr(model_release_command, "choose", lambda *_: "1")
    monkeypatch.setattr("builtins.input", lambda _: "n")
    monkeypatch.setattr(model_release_command, "MODELS_DIR", tmp_path / "models")
    def forbidden(*args, **kwargs):
        pytest.fail("Download should not start after cancellation")
    monkeypatch.setattr(model_release_command, "download_release_asset", forbidden)
    assert model_release_command.run() is False
    assert not (tmp_path / "models").exists()
