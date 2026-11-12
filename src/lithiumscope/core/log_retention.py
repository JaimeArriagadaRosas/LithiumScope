"""Retention policy for LithiumScope diagnostic log files.

Independent of logging handlers and startup. Never deletes .gitkeep,
non-log files or files newer than their configured retention age.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path


def cleanup_expired_logs(
    root: Path, retention_days: int,
    error_retention_days: int, now: datetime | None = None,
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
