from __future__ import annotations

import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import sys
import threading
from typing import Iterator
import warnings

from lithiumscope.core.config import load_config
from lithiumscope.core.paths import LOGS_DIR, PROJECT_ROOT, ensure_runtime_directories

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_CONSOLE_FORMAT = "%(levelname)s | %(name)s | %(message)s"
_CONFIGURED = False
_CURRENT_SESSION_LOG: Path | None = None
_CURRENT_ERROR_LOG: Path | None = None
_SESSION_HANDLER: logging.Handler | None = None
_ERROR_HANDLER: logging.Handler | None = None
_WARNING_CAPTURE_INSTALLED = False
_ORIGINAL_SHOWWARNING = warnings.showwarning
_WARNING_COUNTS: dict[tuple[str, str, str], int] = {}
_WARNING_LOCK = threading.RLock()
_ERROR_COUNT = 0


class ConciseConsoleFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        exc_info = record.exc_info
        exc_text = record.exc_text
        record.exc_info = None
        record.exc_text = None
        try:
            return super().format(record)
        finally:
            record.exc_info = exc_info
            record.exc_text = exc_text


class CoordinatedConsoleHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            from lithiumscope.runtime.console_status import console_message
            console_message(self.format(record))
        except Exception:
            self.handleError(record)


class _ErrorCountingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        global _ERROR_COUNT
        _ERROR_COUNT += 1
        return True


def _log_root() -> Path:
    override = os.environ.get("LITHIUMSCOPE_LOG_DIR")
    if override:
        return Path(override)
    if "pytest" in sys.modules:
        return PROJECT_ROOT / ".test_logs" / f"pid_{os.getpid()}"
    return LOGS_DIR


def cleanup_expired_logs(root: Path, retention_days: int, error_retention_days: int, now: datetime | None = None) -> int:
    if not root.exists():
        return 0
    now = now or datetime.now(timezone.utc)
    removed = 0
    for path in root.rglob("*"):
        if not path.is_file() or path.name == ".gitkeep":
            continue
        if path.suffix.lower() not in {".log", ".json"}:
            continue
        retention = error_retention_days if "errors" in path.parts else retention_days
        modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        if (now - modified).total_seconds() / 86400 <= retention:
            continue
        try:
            path.unlink()
            removed += 1
        except OSError:
            pass
    return removed


def _handler_is_attached(root: logging.Logger, handler: logging.Handler | None) -> bool:
    if handler is None or handler not in root.handlers:
        return False
    stream = getattr(handler, "stream", None)
    return stream is None or not getattr(stream, "closed", False)


def _flush_handlers(root: logging.Logger) -> None:
    for handler in tuple(root.handlers):
        try:
            handler.flush()
        except (OSError, ValueError):
            pass


def _teardown_handlers(root: logging.Logger) -> None:
    for handler in tuple(root.handlers):
        if not getattr(handler, "_lithiumscope_owned", False):
            continue
        root.removeHandler(handler)
        try:
            handler.flush()
        except (OSError, ValueError):
            pass
        try:
            handler.close()
        except (OSError, ValueError):
            pass


def configure_logging() -> logging.Logger:
    global _CONFIGURED, _CURRENT_SESSION_LOG, _CURRENT_ERROR_LOG, _SESSION_HANDLER, _ERROR_HANDLER, _ERROR_COUNT
    ensure_runtime_directories()
    root = logging.getLogger("lithiumscope")
    if _CONFIGURED and _handler_is_attached(root, _SESSION_HANDLER) and _handler_is_attached(root, _ERROR_HANDLER):
        return root
    if _CONFIGURED:
        _teardown_handlers(root)
    else:
        _ERROR_COUNT = 0
        with _WARNING_LOCK:
            _WARNING_COUNTS.clear()

    cfg = load_config("logging").get("logging", {})
    file_level = getattr(logging, str(cfg.get("file_level", "INFO")).upper(), logging.INFO)
    console_level = getattr(logging, str(cfg.get("console_level", "ERROR")).upper(), logging.ERROR)
    max_bytes = int(cfg.get("max_bytes", 5 * 1024 * 1024))
    backup_count = int(cfg.get("backup_count", 5))
    retention_days = int(cfg.get("retention_days", 14))
    error_retention_days = int(cfg.get("error_retention_days", 30))

    log_root = _log_root()
    session_dir = log_root / "sessions"
    error_dir = log_root / "errors"
    session_dir.mkdir(parents=True, exist_ok=True)
    error_dir.mkdir(parents=True, exist_ok=True)
    removed = cleanup_expired_logs(log_root, retention_days, error_retention_days)

    session_stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    _CURRENT_SESSION_LOG = session_dir / f"session_{session_stamp}.log"
    _CURRENT_ERROR_LOG = error_dir / f"errors_{session_stamp}.log"

    root.setLevel(logging.DEBUG)
    root.propagate = False
    formatter = logging.Formatter(_LOG_FORMAT)

    session_handler = RotatingFileHandler(_CURRENT_SESSION_LOG, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8", delay=False)
    session_handler._lithiumscope_owned = True
    session_handler.setFormatter(formatter)
    session_handler.setLevel(file_level)
    root.addHandler(session_handler)

    error_handler = RotatingFileHandler(_CURRENT_ERROR_LOG, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8", delay=True)
    error_handler._lithiumscope_owned = True
    error_handler.setFormatter(formatter)
    error_handler.setLevel(logging.ERROR)
    error_handler.addFilter(_ErrorCountingFilter())
    root.addHandler(error_handler)

    if bool(cfg.get("console", False)):
        console = CoordinatedConsoleHandler()
        console._lithiumscope_owned = True
        console.setFormatter(ConciseConsoleFormatter(_CONSOLE_FORMAT))
        console.setLevel(console_level)
        root.addHandler(console)

    _SESSION_HANDLER = session_handler
    _ERROR_HANDLER = error_handler
    _CONFIGURED = True

    root.info(
        "SESSION_START pid=%d session=%s retention_days=%d error_retention_days=%d removed_expired=%d",
        os.getpid(),
        _CURRENT_SESSION_LOG,
        retention_days,
        error_retention_days,
        removed,
    )
    _flush_handlers(root)
    if not _CURRENT_SESSION_LOG.exists() or _CURRENT_SESSION_LOG.stat().st_size <= 0:
        raise RuntimeError(f"El log de sesión no pudo inicializarse correctamente: {_CURRENT_SESSION_LOG}")
    return root


def install_warning_capture() -> None:
    global _WARNING_CAPTURE_INSTALLED
    if _WARNING_CAPTURE_INSTALLED:
        return

    def _showwarning(message, category, filename, lineno, file=None, line=None):
        key = (category.__name__, str(message), str(filename))
        with _WARNING_LOCK:
            count = _WARNING_COUNTS.get(key, 0) + 1
            _WARNING_COUNTS[key] = count
        logger = get_logger("warnings")
        if count == 1:
            logger.warning("%s at %s:%d | %s", category.__name__, filename, lineno, message)
        else:
            logger.debug("Repeated warning #%d: %s | %s", count, category.__name__, message)

    warnings.showwarning = _showwarning
    _WARNING_CAPTURE_INSTALLED = True


def warning_summary() -> dict[str, int]:
    with _WARNING_LOCK:
        unique = len(_WARNING_COUNTS)
        total = sum(_WARNING_COUNTS.values())
    return {"unique": unique, "total": total, "repeated": max(0, total - unique)}


def error_count() -> int:
    return int(_ERROR_COUNT)


def logging_summary() -> dict[str, object]:
    return {
        "session_log": str(_CURRENT_SESSION_LOG) if _CURRENT_SESSION_LOG else None,
        "session_log_exists": bool(_CURRENT_SESSION_LOG and _CURRENT_SESSION_LOG.exists()),
        "session_log_size": _CURRENT_SESSION_LOG.stat().st_size if _CURRENT_SESSION_LOG and _CURRENT_SESSION_LOG.exists() else 0,
        "error_log": str(_CURRENT_ERROR_LOG) if _CURRENT_ERROR_LOG else None,
        "error_log_exists": bool(_CURRENT_ERROR_LOG and _CURRENT_ERROR_LOG.exists()),
        "error_log_size": _CURRENT_ERROR_LOG.stat().st_size if _CURRENT_ERROR_LOG and _CURRENT_ERROR_LOG.exists() else 0,
        "warnings": warning_summary(),
        "errors": error_count(),
    }


def current_session_log_path() -> Path:
    configure_logging()
    assert _CURRENT_SESSION_LOG is not None
    return _CURRENT_SESSION_LOG


def current_error_log_path() -> Path:
    configure_logging()
    assert _CURRENT_ERROR_LOG is not None
    return _CURRENT_ERROR_LOG


def flush_logging() -> None:
    if _CONFIGURED:
        _flush_handlers(logging.getLogger("lithiumscope"))


def finalize_logging() -> None:
    global _CONFIGURED, _WARNING_CAPTURE_INSTALLED
    if not _CONFIGURED:
        return
    root = logging.getLogger("lithiumscope")
    summary = warning_summary()
    root.info(
        "SESSION_END warnings_unique=%d warnings_total=%d errors=%d",
        summary["unique"],
        summary["total"],
        error_count(),
    )
    _flush_handlers(root)
    _teardown_handlers(root)
    _CONFIGURED = False
    if _WARNING_CAPTURE_INSTALLED:
        warnings.showwarning = _ORIGINAL_SHOWWARNING
        _WARNING_CAPTURE_INSTALLED = False


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(f"lithiumscope.{name}")


@contextmanager
def run_log(category: str, name: str) -> Iterator[Path]:
    logger = configure_logging()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    directory = _log_root() / category
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}_{timestamp}.log"
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    try:
        logger.info("Run log started: %s", path)
        yield path
    finally:
        logger.info("Run log finished: %s", path)
        logger.removeHandler(handler)
        handler.flush()
        handler.close()
