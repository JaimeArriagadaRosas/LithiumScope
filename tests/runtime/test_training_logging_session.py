import json

import pytest

from lithiumscope.core import training_logs


def test_training_session_records_console_and_success(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(training_logs, "LOGS_DIR", tmp_path)
    monkeypatch.setattr(training_logs, "configure_logging", lambda: None)
    with training_logs.training_session("3") as path:
        print("fold complete")
        assert (path / "console.log").exists()
    status = json.loads((path / "status.json").read_text(encoding="utf-8"))
    assert status["state"] == "completed"
    assert "fold complete" in (path / "console.log").read_text(encoding="utf-8")
    assert (path / "events.log").exists()


def test_training_session_preserves_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(training_logs, "LOGS_DIR", tmp_path)
    monkeypatch.setattr(training_logs, "configure_logging", lambda: None)
    with pytest.raises(RuntimeError, match="failed preparation"):
        with training_logs.training_session("3") as path:
            print("starting M1")
            raise RuntimeError("failed preparation")
    status = json.loads((path / "status.json").read_text(encoding="utf-8"))
    assert status["state"] == "failed"
    assert "failed preparation" in (
        path / "failure_traceback.txt"
    ).read_text(encoding="utf-8")
