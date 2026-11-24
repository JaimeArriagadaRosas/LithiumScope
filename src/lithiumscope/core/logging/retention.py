"""Conservative retention preserving active and scientifically linked sessions."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path


def _active(marker: Path) -> bool:
    try:
        import psutil
        payload = json.loads(marker.read_text(encoding="utf-8"))
        process = psutil.Process(int(payload["pid"]))
        recorded = payload.get("process_start")
        return recorded is None or abs(process.create_time() - recorded) < 1
    except ImportError:
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return True
    except Exception as exc:
        # Only a positively absent process is considered inactive.
        return type(exc).__name__ != "NoSuchProcess"


def cleanup_expired_logs(root: Path, retention_days: int, error_retention_days: int,
                         now: datetime | None = None) -> int:
    if not root.exists():
        return 0
    now = now or datetime.now(timezone.utc)
    protected = set()
    for session in (root / "sessions").glob("*"):
        if not session.is_dir():
            continue
        if any(_active(marker) for marker in session.rglob(".active.json")):
            protected.add(session)
        # Scientific run links and statuses are evidence, not disposable logs.
        if any(session.rglob("runs.json")) or any(session.rglob("status.json")):
            protected.add(session)
    removed = 0
    for path in root.rglob("*"):
        if not path.is_file() or path.name == ".gitkeep" or path.name == ".active.json":
            continue
        if any(path.is_relative_to(session) for session in protected):
            continue
        if path.suffix.lower() not in {".log", ".jsonl"}:
            continue
        retention = error_retention_days if "errors" in path.parts else retention_days
        try:
            modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
            if (now - modified).total_seconds() / 86400 > retention:
                path.unlink()
                removed += 1
        except OSError:
            pass
    return removed
