from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from lithiumscope.core.logger import cleanup_expired_logs


def test_concurrent_operations_share_session_but_not_operation_logs(tmp_path):
    env = os.environ.copy()
    env["LITHIUMSCOPE_LOG_DIR"] = str(tmp_path)
    env.pop("LITHIUMSCOPE_SESSION_DIR", None)
    env.pop("LITHIUMSCOPE_WORKER_SESSION_DIR", None)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2] / "src")
    code = """
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from lithiumscope.core.logger import get_logger, run_log, finalize_logging
barrier = Barrier(2)
def work(name):
    with run_log("prediction", name):
        barrier.wait(timeout=10)
        get_logger("probe").info("message-" + name)
with ThreadPoolExecutor(max_workers=2) as executor:
    list(executor.map(work, ["alpha", "beta"]))
finalize_logging()
"""
    subprocess.run([sys.executable, "-c", code], env=env, check=True, timeout=30)
    sessions = list((tmp_path / "sessions").glob("session_*"))
    assert len(sessions) == 1 and sessions[0].is_dir()
    session = sessions[0]
    general = [json.loads(line) for line in (session / "session.log").read_text(encoding="utf-8").splitlines()]
    assert {row["session_id"] for row in general} == {session.name}
    operations = list((session / "prediction").glob("*.log"))
    assert len(operations) == 2
    for path in operations:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        ids = {row["operation_id"] for row in rows}
        assert len(ids) == 1 and None not in ids
        messages = [r["message"] for r in rows]
        assert sum(any("message-" + name in m for m in messages) for name in ("alpha", "beta")) == 1
        assert path.with_suffix(".resources.jsonl").is_file()


def test_retention_preserves_training_session_evidence(tmp_path):
    session = tmp_path / "sessions" / "session_old"
    training = session / "training" / "run"
    training.mkdir(parents=True)
    (training / "status.json").write_text('{"state":"interrupted"}')
    path = training / "events.log"
    path.write_text("evidence")
    now = datetime.now(timezone.utc)
    old = (now - timedelta(days=60)).timestamp()
    os.utime(path, (old, old))
    cleanup_expired_logs(tmp_path, 14, 30, now)
    assert path.exists() and (training / "status.json").exists()
