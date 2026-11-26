"""Small pytest observer. Does not alter collection, test execution or outcomes."""
from __future__ import annotations

import json
import os
from pathlib import Path


def _emit(kind, **data):
    destination = os.environ.get("LITHIUMSCOPE_PYTEST_PROGRESS")
    if not destination:
        return
    # Progress is advisory. A reporting failure must never fail a scientific test.
    try:
        with Path(destination).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"kind": kind, **data}, ensure_ascii=True) + "\n")
    except OSError:
        pass


def pytest_sessionstart(session):
    _emit("collecting")


def pytest_collection_finish(session):
    _emit("collected", total=len(session.items))


def pytest_runtest_logstart(nodeid, location):
    _emit("start", nodeid=nodeid)


def pytest_runtest_logreport(report):
    if report.failed or report.skipped or report.when == "call":
        _emit("report", nodeid=report.nodeid, phase=report.when,
              outcome=report.outcome)


def pytest_runtest_logfinish(nodeid, location):
    _emit("finish", nodeid=nodeid)


def pytest_sessionfinish(session, exitstatus):
    _emit("session_finished", exit_code=int(exitstatus))
