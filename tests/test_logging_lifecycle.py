from pathlib import Path
import os
import subprocess
import sys


def _run_logging_probe(tmp_path: Path, code: str):
    env = os.environ.copy()
    env["LITHIUMSCOPE_LOG_DIR"] = str(tmp_path)
    return subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True, env=env, timeout=20)


def test_session_log_is_nonempty_and_errors_are_lazy(tmp_path):
    _run_logging_probe(
        tmp_path,
        "from lithiumscope.core.logger import configure_logging,get_logger,finalize_logging;"
        "configure_logging();get_logger('probe').info('hello');finalize_logging()",
    )
    sessions = list((tmp_path / "sessions").glob("session_*.log"))
    errors = list((tmp_path / "errors").glob("errors_*.log"))
    assert len(sessions) == 1
    assert sessions[0].stat().st_size > 0
    text = sessions[0].read_text(encoding="utf-8")
    assert "SESSION_START" in text
    assert "SESSION_END" in text
    assert errors == []


def test_error_log_is_created_on_first_error(tmp_path):
    _run_logging_probe(
        tmp_path,
        "from lithiumscope.core.logger import configure_logging,get_logger,finalize_logging;"
        "configure_logging();get_logger('probe').error('boom');finalize_logging()",
    )
    errors = list((tmp_path / "errors").glob("errors_*.log"))
    assert len(errors) == 1
    assert errors[0].stat().st_size > 0
    assert "boom" in errors[0].read_text(encoding="utf-8")
