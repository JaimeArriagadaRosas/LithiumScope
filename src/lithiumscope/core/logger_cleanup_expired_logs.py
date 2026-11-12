"""Separated cleanup_expired_logs responsibility; preserve prior public API."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lithiumscope.core.logger import Path, datetime

def cleanup_expired_logs(root: Path, retention_days: int, error_retention_days: int, now: datetime | None = None) -> int:
    """Compatible public facade; housekeeping is separated from log handlers."""
    from lithiumscope.core.logger import Path, datetime
    from lithiumscope.core.log_retention import cleanup_expired_logs as cleanup
    return cleanup(root, retention_days, error_retention_days, now=now)
