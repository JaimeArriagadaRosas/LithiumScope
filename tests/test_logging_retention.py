from datetime import datetime, timedelta, timezone
import os

from lithiumscope.core.logger import cleanup_expired_logs


def test_old_logs_expire_but_recent_logs_remain(tmp_path):
    old = tmp_path / "training" / "old.log"
    recent = tmp_path / "training" / "recent.log"
    error = tmp_path / "errors" / "errors.log"
    for path in (old, recent, error):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x", encoding="utf-8")

    now = datetime.now(timezone.utc)
    old_time = (now - timedelta(days=20)).timestamp()
    recent_time = (now - timedelta(days=2)).timestamp()
    error_time = (now - timedelta(days=20)).timestamp()
    os.utime(old, (old_time, old_time))
    os.utime(recent, (recent_time, recent_time))
    os.utime(error, (error_time, error_time))

    removed = cleanup_expired_logs(
        tmp_path,
        retention_days=14,
        error_retention_days=30,
        now=now,
    )

    assert removed == 1
    assert not old.exists()
    assert recent.exists()
    assert error.exists()
