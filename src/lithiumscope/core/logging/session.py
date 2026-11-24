"""Process-safe logging session owner. Existing public API remains compatible."""
from __future__ import annotations

import atexit
import json
import logging
import os
from pathlib import Path
import sys
import threading
import warnings

from lithiumscope.core.config import load_config
from lithiumscope.core.paths import CONFIG_DIR, LOGS_DIR, PROJECT_ROOT
from lithiumscope.core.logging.context import unique_id
from lithiumscope.core.logging.formatters import ConciseConsoleFormatter, StructuredFormatter
from lithiumscope.core.logging.handlers import CoordinatedConsoleHandler, LazyRotatingFileHandler, flush_handlers, teardown_handlers
from lithiumscope.core.logging.retention import cleanup_expired_logs

_LOCK = threading.RLock()
_WARNING_LOCK = threading.RLock()
_ERROR_LOCK = threading.Lock()
_CONFIGURED = False
_SESSION_ID: str | None = None
_SESSION_DIR: Path | None = None
_CURRENT_SESSION_LOG: Path | None = None
_CURRENT_ERROR_LOG: Path | None = None
_SESSION_HANDLER: logging.Handler | None = None
_ERROR_HANDLER: logging.Handler | None = None
_WARNING_CAPTURE_INSTALLED = False
_ORIGINAL_SHOWWARNING = warnings.showwarning
_WARNING_COUNTS: dict[tuple[str, str, str], int] = {}
_ERROR_COUNT = 0


class _ErrorCountingFilter(logging.Filter):
    def filter(self, record):
        global _ERROR_COUNT
        with _ERROR_LOCK:
            _ERROR_COUNT += 1
        return True


def _log_root() -> Path:
    override = os.environ.get("LITHIUMSCOPE_LOG_DIR")
    if override:
        return Path(override)
    if "pytest" in sys.modules:
        return PROJECT_ROOT / ".test_logs" / f"pid_{os.getpid()}"
    return LOGS_DIR


def _new_session() -> tuple[str, Path]:
    parent = os.environ.get("LITHIUMSCOPE_SESSION_DIR")
    if parent and not os.environ.get("LITHIUMSCOPE_LOG_DIR"):
        root = Path(parent)
        return root.name, root / "tests" / unique_id(f"process_{os.getpid()}")
    identity = unique_id("session")
    return identity, _log_root() / "sessions" / identity


def _attached(root, handler):
    return handler is not None and handler in root.handlers and (
        getattr(handler, "stream", None) is None or not handler.stream.closed
    )


def configure_logging() -> logging.Logger:
    global _CONFIGURED, _SESSION_ID, _SESSION_DIR, _CURRENT_SESSION_LOG, _CURRENT_ERROR_LOG
    global _SESSION_HANDLER, _ERROR_HANDLER, _ERROR_COUNT
    with _LOCK:
        root = logging.getLogger("lithiumscope")
        if _CONFIGURED and _attached(root, _SESSION_HANDLER) and _attached(root, _ERROR_HANDLER):
            return root
        if _CONFIGURED:
            teardown_handlers(root)
            if _SESSION_DIR:
                (_SESSION_DIR / ".active.json").unlink(missing_ok=True)
        with _ERROR_LOCK:
            _ERROR_COUNT = 0
        with _WARNING_LOCK:
            _WARNING_COUNTS.clear()
        # Diagnostics must never install missing configs merely to create a log.
        cfg = load_config("logging").get("logging", {}) if (CONFIG_DIR / "logging.yaml").is_file() else {}
        file_level = getattr(logging, str(cfg.get("file_level", "INFO")).upper(), logging.INFO)
        console_level = getattr(logging, str(cfg.get("console_level", "ERROR")).upper(), logging.ERROR)
        max_bytes = int(cfg.get("max_bytes", 5 * 1024 * 1024))
        backups = int(cfg.get("backup_count", 5))
        retention = int(cfg.get("retention_days", 14))
        error_retention = int(cfg.get("error_retention_days", 30))
        _SESSION_ID, _SESSION_DIR = _new_session()
        _SESSION_DIR.mkdir(parents=True, exist_ok=False)
        process_start = None
        try:
            import psutil
            process_start = psutil.Process().create_time()
        except (ImportError, OSError):
            pass
        (_SESSION_DIR / ".active.json").write_text(json.dumps({"pid": os.getpid(), "process_start": process_start}), encoding="utf-8")
        if os.environ.get("LITHIUMSCOPE_SESSION_DIR") and not os.environ.get("LITHIUMSCOPE_LOG_DIR"):
            removed = 0
        else:
            removed = cleanup_expired_logs(_log_root(), retention, error_retention)
        _CURRENT_SESSION_LOG = _SESSION_DIR / "session.log"
        _CURRENT_ERROR_LOG = _SESSION_DIR / "errors" / "errors.log"
        root.setLevel(logging.DEBUG)
        root.propagate = False
        formatter = StructuredFormatter(_SESSION_ID)
        session = LazyRotatingFileHandler(_CURRENT_SESSION_LOG, maxBytes=max_bytes,
                                          backupCount=backups, encoding="utf-8")
        session.setFormatter(formatter)
        session.setLevel(file_level)
        session._lithiumscope_owned = True
        root.addHandler(session)
        error = LazyRotatingFileHandler(_CURRENT_ERROR_LOG, maxBytes=max_bytes,
                                        backupCount=backups, encoding="utf-8", delay=True)
        error.setFormatter(formatter)
        error.setLevel(logging.ERROR)
        error.addFilter(_ErrorCountingFilter())
        error._lithiumscope_owned = True
        root.addHandler(error)
        if bool(cfg.get("console", False)):
            console = CoordinatedConsoleHandler()
            console.setFormatter(ConciseConsoleFormatter("%(levelname)s | %(name)s | %(message)s"))
            console.setLevel(console_level)
            console._lithiumscope_owned = True
            root.addHandler(console)
        _SESSION_HANDLER, _ERROR_HANDLER, _CONFIGURED = session, error, True
        root.info("SESSION_START pid=%d removed_expired=%d", os.getpid(), removed)
        flush_handlers(root)
        atexit.register(finalize_logging)
        return root


def current_session_directory() -> Path:
    configure_logging()
    assert _SESSION_DIR is not None
    return _SESSION_DIR


def current_session_id() -> str:
    configure_logging()
    assert _SESSION_ID is not None
    return _SESSION_ID


def current_session_log_path() -> Path:
    configure_logging()
    assert _CURRENT_SESSION_LOG is not None
    return _CURRENT_SESSION_LOG


def current_error_log_path() -> Path:
    configure_logging()
    assert _CURRENT_ERROR_LOG is not None
    return _CURRENT_ERROR_LOG


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(f"lithiumscope.{name}")


def install_warning_capture() -> None:
    global _WARNING_CAPTURE_INSTALLED
    if _WARNING_CAPTURE_INSTALLED:
        return

    def capture(message, category, filename, lineno, file=None, line=None):
        key = (category.__name__, str(message), str(filename))
        with _WARNING_LOCK:
            count = _WARNING_COUNTS.get(key, 0) + 1
            _WARNING_COUNTS[key] = count
        log = get_logger("warnings")
        if count == 1:
            log.warning("%s at %s:%d | %s", category.__name__, filename, lineno, message)
        else:
            log.debug("Repeated warning #%d: %s | %s", count, category.__name__, message)
    warnings.showwarning = capture
    _WARNING_CAPTURE_INSTALLED = True


def warning_summary() -> dict:
    with _WARNING_LOCK:
        unique, total = len(_WARNING_COUNTS), sum(_WARNING_COUNTS.values())
    return {"unique": unique, "total": total, "repeated": max(0, total - unique)}


def error_count() -> int:
    with _ERROR_LOCK:
        return _ERROR_COUNT


def logging_summary() -> dict:
    def size(path):
        return path.stat().st_size if path and path.is_file() else 0
    return {"session_log": str(_CURRENT_SESSION_LOG) if _CURRENT_SESSION_LOG else None,
            "session_log_exists": bool(_CURRENT_SESSION_LOG and _CURRENT_SESSION_LOG.is_file()),
            "session_log_size": size(_CURRENT_SESSION_LOG),
            "error_log": str(_CURRENT_ERROR_LOG) if _CURRENT_ERROR_LOG else None,
            "error_log_exists": bool(_CURRENT_ERROR_LOG and _CURRENT_ERROR_LOG.is_file()),
            "error_log_size": size(_CURRENT_ERROR_LOG), "warnings": warning_summary(),
            "errors": error_count()}


def flush_logging() -> None:
    if _CONFIGURED:
        flush_handlers(logging.getLogger("lithiumscope"))


def finalize_logging() -> None:
    global _CONFIGURED, _WARNING_CAPTURE_INSTALLED
    with _LOCK:
        if not _CONFIGURED:
            return
        root = logging.getLogger("lithiumscope")
        summary = warning_summary()
        root.info("SESSION_END warnings_unique=%d warnings_total=%d errors=%d",
                  summary["unique"], summary["total"], error_count())
        flush_handlers(root)
        teardown_handlers(root)
        if _SESSION_DIR:
            (_SESSION_DIR / ".active.json").unlink(missing_ok=True)
        _CONFIGURED = False
        if _WARNING_CAPTURE_INSTALLED:
            warnings.showwarning = _ORIGINAL_SHOWWARNING
            _WARNING_CAPTURE_INSTALLED = False


def run_log(category: str, name: str):
    from lithiumscope.core.logging.operations import run_log as operation
    return operation(category, name)
