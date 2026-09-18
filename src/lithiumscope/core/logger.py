from __future__ import annotations

import logging
from contextlib import contextmanager
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Iterator

from lithiumscope.core.config import load_config
from lithiumscope.core.paths import LOGS_DIR, ensure_runtime_directories

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_CONFIGURED = False


def configure_logging() -> logging.Logger:
    global _CONFIGURED
    ensure_runtime_directories()
    root = logging.getLogger("lithiumscope")
    if _CONFIGURED:
        return root

    cfg = load_config("logging").get("logging", {})
    level = getattr(logging, str(cfg.get("level", "INFO")).upper(), logging.INFO)
    max_bytes = int(cfg.get("max_bytes", 5 * 1024 * 1024))
    backup_count = int(cfg.get("backup_count", 5))

    root.setLevel(level)
    root.propagate = False
    formatter = logging.Formatter(_LOG_FORMAT)

    app_handler = RotatingFileHandler(
        LOGS_DIR / "lithiumscope.log",
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    app_handler.setFormatter(formatter)
    app_handler.setLevel(level)
    root.addHandler(app_handler)

    error_handler = RotatingFileHandler(
        LOGS_DIR / "errors" / "errors.log",
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    error_handler.setFormatter(formatter)
    error_handler.setLevel(logging.ERROR)
    root.addHandler(error_handler)

    if bool(cfg.get("console", True)):
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        console.setLevel(level)
        root.addHandler(console)

    _CONFIGURED = True
    root.info("Centralized logging initialized")
    return root


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(f"lithiumscope.{name}")


@contextmanager
def run_log(category: str, name: str) -> Iterator[Path]:
    """Attach a temporary per-run log while keeping the central log active."""
    logger = configure_logging()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    directory = LOGS_DIR / category
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}_{timestamp}.log"

    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    handler.setLevel(logger.level)
    logger.addHandler(handler)
    try:
        logger.info("Run log started: %s", path)
        yield path
    finally:
        logger.info("Run log finished: %s", path)
        logger.removeHandler(handler)
        handler.close()
