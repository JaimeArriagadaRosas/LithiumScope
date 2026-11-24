"""Owned handlers and lazy error files."""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from lithiumscope.core.logging.context import operation_id


class LazyRotatingFileHandler(RotatingFileHandler):
    def _open(self):
        Path(self.baseFilename).parent.mkdir(parents=True, exist_ok=True)
        return super()._open()


class CoordinatedConsoleHandler(logging.Handler):
    def emit(self, record):
        try:
            from lithiumscope.runtime.console_status import console_message
            console_message(self.format(record))
        except Exception:
            self.handleError(record)


class OperationFilter(logging.Filter):
    def __init__(self, identity: str):
        super().__init__()
        self.identity = identity

    def filter(self, record):
        return operation_id.get() == self.identity


def flush_handlers(root: logging.Logger) -> None:
    for handler in tuple(root.handlers):
        try:
            handler.flush()
        except (OSError, ValueError):
            pass


def teardown_handlers(root: logging.Logger) -> None:
    for handler in tuple(root.handlers):
        if not getattr(handler, "_lithiumscope_owned", False):
            continue
        root.removeHandler(handler)
        try:
            handler.flush()
            handler.close()
        except (OSError, ValueError):
            pass
