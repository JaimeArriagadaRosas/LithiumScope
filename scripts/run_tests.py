"""Run pytest with isolated temporary files and persistent execution evidence."""
from __future__ import annotations

import datetime as dt
import json
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import uuid
import xml.etree.ElementTree as ET


if __package__:
    from .test_reporting.architecture import audit
    from .pytest_console import monitor, print_summary, report_details
    from .software_checks import run_preflight, artifact_snapshot, compare_artifacts, validation_outcome
else:
    from test_reporting.architecture import audit
    from pytest_console import monitor, print_summary, report_details
    from software_checks import run_preflight, artifact_snapshot, compare_artifacts, validation_outcome


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


def git_identity() -> dict:
    try:
        def git(*args):
            return subprocess.run(["git", *args], cwd=ROOT, check=True,
                                  capture_output=True, text=True, timeout=30).stdout.strip()
        return {"git_commit": git("rev-parse", "HEAD"),
                "git_branch": git("branch", "--show-current"),
                "tracked_worktree_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
                "untracked_code_present": bool(git("ls-files", "--others", "--exclude-standard",
                                                    "--", "src", "tests", "scripts", "config", "docs/integral-refactor"))}
    except (OSError, subprocess.SubprocessError):
        return {"git_commit": None, "git_branch": None, "tracked_worktree_dirty": None, "untracked_code_present": None}


def coverage_mode(requested: bool) -> str:
    if not requested:
        return "not_requested"
    return "enabled" if importlib.util.find_spec("pytest_cov") is not None else "missing_pytest_cov"


def run(args: list[str]) -> int:
    json_output = "--json" in args
    args = [arg for arg in args if arg != "--json"]
    preflight_requested = "--preflight" in args
    args = [arg for arg in args if arg != "--preflight"]
    coverage_requested = "--coverage" in args
    mode = coverage_mode(coverage_requested)
    coverage = mode == "enabled"
    args = [arg for arg in args if arg != "--coverage"]
    if args and args[0] == "--":
        args = args[1:]
    run_id = new_id()
    session_root = ROOT / "logs" / "sessions"
    inherited = os.environ.get("LITHIUMSCOPE_SESSION_DIR")
    parent = Path(inherited).resolve() if inherited else session_root / new_id()
    if not parent.is_relative_to(session_root.resolve()):
        raise ValueError("Inherited test session must stay under logs/sessions")
    session_id = parent.relative_to(session_root.resolve()).parts[0]
    results = ROOT / "results" / "tests" / run_id
    logs = parent / "tests" / run_id
    results.mkdir(parents=True, exist_ok=False)
    logs.mkdir(parents=True, exist_ok=False)
    events = logs / "events.jsonl"
    report = results / "junit.xml"
    progress_file = logs / "progress.jsonl"
    env = os.environ.copy()
    # Checkout resolution must not inherit an old workspace override.
    env.pop("LITHIUMSCOPE_HOME", None)
    env.pop("PYTHONPATH", None)
    env.pop("LITHIUMSCOPE_LOG_DIR", None)
    env.pop("PYTEST_ADDOPTS", None)
    env["LITHIUMSCOPE_DISABLE_AUTO_INSTALL"] = "1"
    env["LITHIUMSCOPE_TEST_SESSION_ID"] = session_id
    env["LITHIUMSCOPE_PYTEST_PROGRESS"] = str(progress_file)
    env["LITHIUMSCOPE_SESSION_DIR"] = str(parent)
    if coverage:
        env["COVERAGE_FILE"] = str(results / ".coverage")
        args += ["--cov=lithiumscope", f"--cov-report=xml:{results / 'coverage.xml'}", "--cov-report=term-missing"]
    identity = {"session_id": session_id, "run_id": run_id, "operation": "pytest"}
    before = git_identity()
    start = time.monotonic()

    def event(state: str, **details: object) -> None:
        with events.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({**identity, "timestamp": utc_now(),
                                     "state": state, **details}) + "\n")

    code = 1
    error = None
    checks = {}
    protected_before = None
    event("started", arguments=args, coverage_state=mode)
    if mode == "missing_pytest_cov":
        event("warning", reason="pytest_cov_missing; running regression without coverage")
    try:
        if preflight_requested:
            protected_before = artifact_snapshot(ROOT, logs, env, "before")
            checks = run_preflight(ROOT, logs, env, show=not json_output)
            architecture = audit(ROOT)
            (results / "architecture.json").write_text(json.dumps(architecture, indent=2), encoding="utf-8")
            checks["test_architecture"] = {"state": architecture["state"],
                "scope": architecture["scope"], "modules": architecture["modules"],
                "log": (results / "architecture.json").relative_to(ROOT).as_posix()}
            if not json_output:
                print(f"  [{architecture['state']}] Arquitectura de tests: {architecture['modules']} módulos", flush=True)
            event("preflight_finished", checks=checks)
        with tempfile.TemporaryDirectory(prefix="pytest-", dir=results) as tmp:
            command = [sys.executable, "-m", "pytest", "-p", "scripts.pytest_progress", *args,
                       "--basetemp", tmp, "--junitxml", str(report)]
            code = monitor(command, cwd=ROOT, env=env, log=logs / "pytest.log",
                           progress_file=progress_file, show=not json_output)
    except KeyboardInterrupt:
        code = 130
        error = "KeyboardInterrupt"
    except OSError as exc:
        error = str(exc)
    pytest_code = code
    if protected_before is not None:
        checks["protected_artifacts"] = compare_artifacts(
            protected_before, artifact_snapshot(ROOT, logs, env, "after"))
    after = git_identity()
    details = report_details(report, results / "coverage.xml")
    counts = junit_counts(report)
    clean = before == after and before["git_commit"] is not None and before["tracked_worktree_dirty"] is False and before["untracked_code_present"] is False
    if preflight_requested:
        state, code = validation_outcome(pytest_code, checks, details,
            coverage_requested=coverage_requested, clean=clean, counts=counts)
    else:
        state = "passed" if code == 0 else "cancelled" if code == 130 else "failed"
    summary = {**identity, **before, "git_identity_after": after,
               "git_identity_unchanged": before == after and before["git_commit"] is not None,
               "state": state, "exit_code": code,
               "duration_seconds": round(time.monotonic() - start, 3),
               "counts": counts, "error": error, "pytest_exit_code": pytest_code,
               "preflight_requested": preflight_requested, "checks": checks,
               "coverage_requested": coverage_requested, "coverage_state": mode,
               "log": (logs / "pytest.log").relative_to(ROOT).as_posix(),
               "arguments": args,
               "progress_log": progress_file.relative_to(ROOT).as_posix(),
               "details": details}
    temporary = results / ".summary.json.tmp"
    temporary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    temporary.replace(results / "summary.json")
    event(state, exit_code=code, duration_seconds=summary["duration_seconds"], error=error)
    if json_output:
        print(json.dumps(summary, indent=2))
    else:
        print_summary(summary, results.relative_to(ROOT))
    return code


if __name__ == "__main__":
    arguments = sys.argv[1:]
    if arguments and arguments[0] == "--":
        arguments = arguments[1:]
    raise SystemExit(run(arguments or ["-q"]))
