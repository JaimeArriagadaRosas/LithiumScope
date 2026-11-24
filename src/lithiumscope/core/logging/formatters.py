"""Structured technical logs and concise console messages."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import logging

from lithiumscope.core.logging.context import operation_id, operation_category


class StructuredFormatter(logging.Formatter):
    def __init__(self, session_id: str):
        super().__init__()
        self.session_id = session_id

    def format(self, record):
        payload = {
            "timestamp_utc": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "session_id": self.session_id, "operation_id": operation_id.get(),
            "operation_category": operation_category.get(), "level": record.levelname,
            "logger": record.name, "module": record.module, "file": record.pathname,
            "function": record.funcName, "line": record.lineno, "pid": record.process,
            "thread": record.thread, "message": record.getMessage(),
        }
        for name in ("algorithm", "fold", "trial", "state", "stage", "run_id", "error_id", "duration_seconds", "config", "resources"):
            if hasattr(record, name):
                payload[name] = getattr(record, name)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


class ConciseConsoleFormatter(logging.Formatter):
    def format(self, record):
        exc_info, exc_text = record.exc_info, record.exc_text
        record.exc_info = record.exc_text = None
        try:
            return super().format(record)
        finally:
            record.exc_info, record.exc_text = exc_info, exc_text
