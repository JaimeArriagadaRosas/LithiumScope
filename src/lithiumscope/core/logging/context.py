"""Execution identity shared by structured handlers."""
from __future__ import annotations

from contextvars import ContextVar
from contextlib import contextmanager
from datetime import datetime, timezone
from uuid import uuid4


def unique_id(prefix: str) -> str:
    return f"{prefix}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S_%fZ}_{uuid4().hex[:10]}"


operation_id: ContextVar[str | None] = ContextVar("operation_id", default=None)
operation_category: ContextVar[str | None] = ContextVar("operation_category", default=None)
scientific_context: ContextVar[dict | None] = ContextVar("scientific_context", default=None)


@contextmanager
def scientific_log_context(**fields):
    """Bind nested scientific event identifiers without process-global mutation."""
    allowed={"algorithm","fold","trial","run_id","stage","variant_id"}
    if set(fields)-allowed:
        raise ValueError("Unknown scientific log context field")
    token=scientific_context.set({**(scientific_context.get() or {}),**fields})
    try:
        yield
    finally:
        scientific_context.reset(token)
