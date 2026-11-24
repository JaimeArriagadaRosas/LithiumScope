"""Execution identity shared by structured handlers."""
from __future__ import annotations

from contextvars import ContextVar
from datetime import datetime, timezone
from uuid import uuid4


def unique_id(prefix: str) -> str:
    return f"{prefix}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S_%fZ}_{uuid4().hex[:10]}"


operation_id: ContextVar[str | None] = ContextVar("operation_id", default=None)
operation_category: ContextVar[str | None] = ContextVar("operation_category", default=None)
