"""Read-only preflight and bounded artifact checks for the full software suite."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time


def _command(root: Path, logs: Path, name: str, arguments: list[str], env: dict,
             *, show: bool, timeout: int = 120) -> dict:
    log = logs / f"{name}.log"
    start = time.monotonic()
    if show:
        print(f"  [PREVIO] {name}...", flush=True)
    code, error = None, None
    try:
        with log.open("wb") as stream:
            completed = subprocess.run([sys.executable, *arguments], cwd=root, env=env,
                                       stdout=stream, stderr=subprocess.STDOUT,
                                       check=False, timeout=timeout)
        code = completed.returncode
    except (OSError, subprocess.TimeoutExpired) as exc:
        error = type(exc).__name__
    result = {"state": "OK" if code == 0 else "ERROR", "exit_code": code,
              "duration_seconds": round(time.monotonic() - start, 3),
              "log": log.relative_to(root).as_posix(), "error": error}
    if code != 0 and log.is_file():
        if name == "retired_imports":
            with log.open(encoding="utf-8", errors="replace") as stream:
                findings = [line.strip() for line in stream if line.strip()]
            result["finding_count"] = max(0, len(findings) - 1) if code == 1 and findings and findings[-1].startswith("Retired internal imports") else None
            result["findings"] = findings[:100]
        else:
            from collections import deque
            with log.open(encoding="utf-8", errors="replace") as stream:
                result["log_tail"] = [line.rstrip()[:1000] for line in deque(stream, maxlen=20)]
    if show:
        print(f"  [{result['state']}] {name}: {result['log']}", flush=True)
        if result.get("finding_count") is not None:
            print(f"    Referencias o errores detectados: {result['finding_count']}", flush=True)
    return result


def run_preflight(root: Path, logs: Path, env: dict, *, show: bool = True) -> dict:
    # Always import from the current checkout, never an older installed copy.
    env = {**env, "PYTHONPATH": str(root / "src")}
    env.pop("LITHIUMSCOPE_HOME", None)
    checks = {
        "retired_imports": ["scripts/check_internal_imports.py"],
        "syntax": ["-c", "from pathlib import Path; from lithiumscope.tools.inspection.development.validation import tracked_python, syntax_check; p=Path.cwd(); r=syntax_check(p,tracked_python(p)); print(r); raise SystemExit(r['state']!='OK')"],
        "dependency_graph": ["-c", "import json; from pathlib import Path; from lithiumscope.tools.inspection.development.validation import _check_dependencies; r=_check_dependencies(Path.cwd()); print(json.dumps(r,indent=2)); raise SystemExit(r['state']!='OK')"],
        "code_size": ["scripts/check_code_size.py", "--check"],
        "startup_imports": ["-c", "import lithiumscope.runtime.lifecycle; import lithiumscope.cli.menu; print('Startup imports OK (no launch)')"],
        "lint": ["-m", "ruff", "check", "--no-cache", "main.py", "src", "tests", "scripts"],
    }
    return {name: _command(root, logs, name, args, env, show=show)
            for name, args in checks.items()}


def artifact_snapshot(root: Path, logs: Path, env: dict, phase: str) -> dict:
    env = {**env, "PYTHONPATH": str(root / "src")}
    destination = logs / f"protected_{phase}.json"
    command = ("import json,sys; from pathlib import Path; "
               "from lithiumscope.tools.inspection.development.protected import protected_inventory; "
               "Path(sys.argv[1]).write_text(json.dumps(protected_inventory(Path.cwd())),encoding='utf-8')")
    try:
        result = subprocess.run([sys.executable, "-c", command, str(destination)], cwd=root, env=env,
                                capture_output=True, timeout=180, check=False)
        (logs / f"protected_{phase}.log").write_bytes(result.stdout + result.stderr)
        if result.returncode == 0:
            return {"state": "OK", "inventory": json.loads(destination.read_text(encoding="utf-8"))}
    except (OSError, ValueError, subprocess.TimeoutExpired):
        pass
    return {"state": "ERROR", "inventory": {}}


def compare_artifacts(before: dict, after: dict) -> dict:
    if before["state"] != "OK" or after["state"] != "OK":
        return {"state": "ERROR", "reason": "artifact_inventory_unavailable"}
    a, b = before["inventory"], after["inventory"]
    changes, unavailable = [], []
    for name in sorted(set(a) | set(b)):
        old, new = a.get(name, {"state": "MISSING"}), b.get(name, {"state": "MISSING"})
        if old.get("state") in {"ERROR", "BLOCKED"} or new.get("state") in {"ERROR", "BLOCKED"}:
            unavailable.append(name)
        elif (old.get("state"), old.get("sha256")) != (new.get("state"), new.get("sha256")):
            changes.append(name)
    return {"state": "ERROR" if changes or unavailable else "OK", "checked_paths": len(set(a) | set(b)),
            "changed_paths": changes, "unverified_paths": unavailable,
            "scope": "selected scientific inputs and model artifacts; no raw patch scan"}


def validation_outcome(pytest_code: int, checks: dict, details: dict, *,
                       coverage_requested: bool, clean: bool, counts: dict) -> tuple[str, int]:
    if pytest_code == 130:
        return "cancelled", 130
    if pytest_code != 0:
        return "failed", pytest_code
    if any(record.get("state") != "OK" for record in checks.values()):
        return "failed", 1
    if details.get("collection_errors", 0) or counts.get("failures", 0) or counts.get("errors", 0):
        return "failed", 1
    if not counts or details.get("executed_tests", 0) <= 0:
        return "incomplete", 10
    if not clean or (coverage_requested and details.get("line_coverage_pct") is None):
        return "incomplete", 10
    if counts.get("skipped", 0):
        return "incomplete", 10
    return "passed", 0
