"""Run pytest with isolated temporary files and persistent execution evidence."""
from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import uuid
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def new_id() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex


def junit_counts(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        return {}
    suites = [root] if root.tag == "testsuite" else root.findall("testsuite")
    return {key: sum(int(s.get(key, "0")) for s in suites)
            for key in ("tests", "failures", "errors", "skipped")}


def run(args: list[str]) -> int:
    session_id, run_id = new_id(), new_id()
    results = ROOT / "results" / "tests" / run_id
    logs = ROOT / "logs" / "sessions" / session_id / "tests"
    results.mkdir(parents=True, exist_ok=False)
    logs.mkdir(parents=True, exist_ok=False)
    events = logs / "events.jsonl"
    report = results / "junit.xml"
    env = os.environ.copy()
    # Checkout resolution must not inherit an old workspace override.
    env.pop("LITHIUMSCOPE_HOME", None)
    env.pop("PYTHONPATH", None)
    env["LITHIUMSCOPE_TEST_SESSION_ID"] = session_id
    identity = {"session_id": session_id, "run_id": run_id, "operation": "pytest"}
    start = time.monotonic()

    def event(state: str, **details: object) -> None:
        with events.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({**identity, "timestamp": utc_now(),
                                     "state": state, **details}) + "\n")

    code = 1
    error = None
    event("started", arguments=args)
    try:
        with tempfile.TemporaryDirectory(prefix="pytest-", dir=results) as tmp:
            command = [sys.executable, "-m", "pytest", *args,
                       "--basetemp", tmp, "--junitxml", str(report)]
            with (logs / "pytest.log").open("wb") as output:
                code = subprocess.run(command, cwd=ROOT, env=env,
                                      stdout=output, stderr=subprocess.STDOUT).returncode
    except KeyboardInterrupt:
        code = 130
        error = "KeyboardInterrupt"
    except OSError as exc:
        error = str(exc)
    state = "passed" if code == 0 else "cancelled" if code == 130 else "failed"
    summary = {**identity, "state": state, "exit_code": code,
               "duration_seconds": round(time.monotonic() - start, 3),
               "counts": junit_counts(report), "error": error,
               "log": (logs / "pytest.log").relative_to(ROOT).as_posix(),
               "arguments": args}
    (results / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    event(state, exit_code=code, duration_seconds=summary["duration_seconds"], error=error)
    print(json.dumps(summary, indent=2))
    return code


if __name__ == "__main__":
    arguments = sys.argv[1:]
    if arguments and arguments[0] == "--":
        arguments = arguments[1:]
    raise SystemExit(run(arguments or ["-q"]))
