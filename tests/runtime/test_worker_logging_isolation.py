"""Concurrent workers own files, share a parent session identity and keep context."""
from pathlib import Path
import json
import os
import subprocess
import sys


def test_concurrent_worker_logs_are_disjoint(tmp_path):
    parent=tmp_path/"sessions"/"parent"
    parent.mkdir(parents=True)
    env={**os.environ,"LITHIUMSCOPE_LOG_DIR":str(tmp_path),"LITHIUMSCOPE_WORKER_SESSION_DIR":str(parent)}
    code="""import sys
from lithiumscope.core.logging.session import get_logger,finalize_logging
from lithiumscope.core.logging.context import scientific_log_context
with scientific_log_context(algorithm=sys.argv[1],fold=2,trial=3):
    get_logger('worker').error('worker-event')
finalize_logging()
"""
    workers=[subprocess.Popen([sys.executable,"-c",code,name],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE) for name in ("a","b")]
    for p in workers:
        out,err=p.communicate(timeout=30)
        assert p.returncode==0,err.decode(errors="replace")
    paths=list(parent.glob("workers/*/session.log"))
    assert len(paths)==2
    rows=[]
    for path in paths:
        events=[json.loads(s) for s in path.read_text().splitlines()]
        selected=[r for r in events if r["message"]=="worker-event"]
        assert len(selected)==1
        assert selected[0]["session_id"]=="parent" and selected[0]["fold"]==2 and selected[0]["trial"]==3
        error=path.parent/"errors/errors.log"
        assert [json.loads(s)["error_id"] for s in error.read_text().splitlines()]==[selected[0]["error_id"]]
        rows.extend(selected)
    assert {r["algorithm"] for r in rows}=={"a","b"}
    assert len({r["pid"] for r in rows})==len({r["error_id"] for r in rows})==2
