from __future__ import annotations

import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import sys
from typing import Iterator

from lithiumscope.core.config import load_config
from lithiumscope.core.paths import LOGS_DIR, PROJECT_ROOT, ensure_runtime_directories

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_CONSOLE_FORMAT = "%(levelname)s | %(name)s | %(message)s"
_CONFIGURED = False
_SESSION_STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
_CURRENT_SESSION_LOG: Path | None = None
_CURRENT_ERROR_LOG: Path | None = None


class ConciseConsoleFormatter(logging.Formatter):
    """Keep console messages short while preserving tracebacks in file handlers."""

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


def _log_root() -> Path:
    override = os.environ.get("LITHIUMSCOPE_LOG_DIR")
    if override:
        return Path(override)
    if "pytest" in sys.modules:
        return PROJECT_ROOT / ".pytest_tmp" / "logs"
    return LOGS_DIR


def cleanup_expired_logs(
    root: Path,
    retention_days: int,
    error_retention_days: int,
    now: datetime | None = None,
) -> int:
    if not root.exists():
        return 0

    now = now or datetime.now(timezone.utc)
    removed = 0
    for path in root.rglob("*"):
        if not path.is_file() or path.name == ".gitkeep":
            continue
        if path.suffix.lower() not in {".log", ".json"}:
            continue

        retention = (
            error_retention_days
            if "errors" in path.parts
            else retention_days
        )
        modified = datetime.fromtimestamp(
            path.stat().st_mtime,
            tz=timezone.utc,
        )
        age_days = (now - modified).total_seconds() / 86400
        if age_days <= retention:
            continue

        try:
            path.unlink()
            removed += 1
        except OSError:
            pass
    return removed


def configure_logging() -> logging.Logger:
    global _CONFIGURED, _CURRENT_SESSION_LOG, _CURRENT_ERROR_LOG
    ensure_runtime_directories()
    root = logging.getLogger("lithiumscope")
    if _CONFIGURED:
        return root

    cfg = load_config("logging").get("logging", {})
    file_level = getattr(
        logging,
        str(cfg.get("file_level", cfg.get("level", "INFO"))).upper(),
        logging.INFO,
    )
    console_level = getattr(
        logging,
        str(cfg.get("console_level", "ERROR")).upper(),
        logging.ERROR,
    )
    max_bytes = int(cfg.get("max_bytes", 5 * 1024 * 1024))
    backup_count = int(cfg.get("backup_count", 5))
    retention_days = int(cfg.get("retention_days", 14))
    error_retention_days = int(cfg.get("error_retention_days", 30))

    log_root = _log_root()
    session_dir = log_root / "sessions"
    error_dir = log_root / "errors"
    session_dir.mkdir(parents=True, exist_ok=True)
    error_dir.mkdir(parents=True, exist_ok=True)

    removed = cleanup_expired_logs(
        log_root,
        retention_days=retention_days,
        error_retention_days=error_retention_days,
    )

    _CURRENT_SESSION_LOG = session_dir / f"session_{_SESSION_STAMP}.log"
    _CURRENT_ERROR_LOG = error_dir / f"errors_{_SESSION_STAMP}.log"

    root.setLevel(logging.DEBUG)
    root.propagate = False
    formatter = logging.Formatter(_LOG_FORMAT)

    session_handler = RotatingFileHandler(
        _CURRENT_SESSION_LOG,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    session_handler.setFormatter(formatter)
    session_handler.setLevel(file_level)
    root.addHandler(session_handler)

    error_handler = RotatingFileHandler(
        _CURRENT_ERROR_LOG,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    error_handler.setFormatter(formatter)
    error_handler.setLevel(logging.ERROR)
    root.addHandler(error_handler)

    if bool(cfg.get("console", False)):
        console = logging.StreamHandler()
        console.setFormatter(ConciseConsoleFormatter(_CONSOLE_FORMAT))
        console.setLevel(console_level)
        root.addHandler(console)

    _CONFIGURED = True
    root.info(
        "Centralized logging initialized session=%s errors=%s "
        "retention_days=%d error_retention_days=%d removed_expired=%d",
        _CURRENT_SESSION_LOG,
        _CURRENT_ERROR_LOG,
        retention_days,
        error_retention_days,
        removed,
    )
    return root


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


@contextmanager
def run_log(category: str, name: str) -> Iterator[Path]:
    """Attach a temporary per-run log while keeping the session log active."""
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
        handler.close()
