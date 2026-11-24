"""Unique operation logs and measured resource use, isolated by context."""
from __future__ import annotations

from contextlib import contextmanager
import logging
import re
import time

from lithiumscope.core.logging.context import operation_id, operation_category, unique_id
from lithiumscope.core.logging.formatters import StructuredFormatter
from lithiumscope.core.logging.handlers import OperationFilter
from lithiumscope.core.logging.session import configure_logging, current_session_directory, current_session_id
from lithiumscope.resources.manager import load_policy, resource_operation
from lithiumscope.resources.monitoring.monitor import monitor_operation


def _safe(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", str(value)).strip("_") or "operation"


@contextmanager
def run_log(category: str, name: str):
    logger = configure_logging()
    identity = unique_id(_safe(name))
    directory = current_session_directory() / _safe(category)
    if category == "training":
        from lithiumscope.core.logging.training import active_training_dir
        directory = active_training_dir() or directory
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{identity}.log"
    handler = logging.FileHandler(path, encoding="utf-8", mode="x")
    handler.setFormatter(StructuredFormatter(current_session_id()))
    handler.setLevel(logging.INFO)
    handler.addFilter(OperationFilter(identity))
    token_id = operation_id.set(identity)
    token_category = operation_category.set(category)
    logger.addHandler(handler)
    start = time.monotonic()
    state = "completed"
    try:
        logger.info("OPERATION_START %s", name, extra={"state": "running"})
        monitor_path = path.with_suffix(".resources.jsonl")
        context = resource_operation(monitor_path) if category == "training" else monitor_operation(
            monitor_path, interval_seconds=load_policy().monitor_interval_seconds)
        with context:
            yield path
    except BaseException as exc:
        state = "cancelled" if isinstance(exc, KeyboardInterrupt) else "failed"
        logger.exception("OPERATION_ERROR %s", name, extra={"state": state})
        raise
    finally:
        logger.info("OPERATION_END %s", name, extra={"state": state, "duration_seconds": time.monotonic() - start})
        logger.removeHandler(handler)
        handler.flush()
        handler.close()
        operation_category.reset(token_category)
        operation_id.reset(token_id)
