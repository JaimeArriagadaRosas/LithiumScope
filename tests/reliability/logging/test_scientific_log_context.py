from concurrent.futures import ThreadPoolExecutor
import json
import logging
from datetime import datetime, timedelta, timezone
import os

from lithiumscope.core.logging.context import scientific_log_context
from lithiumscope.core.logging.formatters import StructuredFormatter
from lithiumscope.core.logging.retention import cleanup_expired_logs


def record():
    return logging.LogRecord("lithiumscope.test",logging.ERROR,__file__,10,"error",(),None)


def test_scientific_nested_context_and_error_id_across_handlers():
    a,b=StructuredFormatter("session"),StructuredFormatter("session")
    with scientific_log_context(algorithm="svm",fold=2):
        with scientific_log_context(trial=7,stage="optuna"):
            event=record()
            one,two=json.loads(a.format(event)),json.loads(b.format(event))
            assert one["error_id"]==two["error_id"]
            assert one["algorithm"]=="svm" and one["fold"]==2 and one["trial"]==7
        assert "trial" not in json.loads(a.format(record()))
    assert "algorithm" not in json.loads(a.format(record()))


def test_context_isolated_between_parallel_threads():
    def work(i):
        with scientific_log_context(fold=i,algorithm=f"a{i}"):
            return json.loads(StructuredFormatter("shared").format(record()))
    with ThreadPoolExecutor(max_workers=3) as pool:
        events=list(pool.map(work,range(8)))
    assert [(e["fold"],e["algorithm"]) for e in events]==[(i,f"a{i}") for i in range(8)]
    assert len({e["error_id"] for e in events})==8


def test_retention_handles_rotated_logs_without_erasing_evidence(tmp_path):
    session=tmp_path/"sessions"/"old"
    session.mkdir(parents=True)
    rotated=session/"session.log.1"
    rotated.write_text("expired")
    protected=tmp_path/"sessions"/"scientific"/"training"
    protected.mkdir(parents=True)
    (protected/"status.json").write_text("{}")
    evidence=protected/"events.log.1"
    evidence.write_text("keep")
    now=datetime.now(timezone.utc)
    timestamp=(now-timedelta(days=90)).timestamp()
    for path in (rotated,evidence):
        os.utime(path,(timestamp,timestamp))
    assert cleanup_expired_logs(tmp_path,14,30,now)==1
    assert not rotated.exists() and evidence.exists()
