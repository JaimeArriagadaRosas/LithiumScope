"""Local, temporary validation: syntax, lint, architecture and full pytest evidence."""
from __future__ import annotations

import ast
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from uuid import uuid4
import xml.etree.ElementTree as ET

from lithiumscope.tools.inspection.development.inventory import (
    file_record, git_identity, local_path, test_report,
)


def tracked_python(root: Path) -> list[Path]:
    result = subprocess.run(["git", "ls-files", "-z", "src", "tests", "scripts"],
                            cwd=root, check=True, capture_output=True, timeout=20)
    paths = []
    for name in result.stdout.decode("utf-8").split("\0"):
        if name.endswith(".py"):
            path = local_path(root, name)
            if path.is_file():
                paths.append(path)
    return sorted(set(paths))


def syntax_check(root: Path, paths: list[Path]) -> dict:
    failures = []
    for path in paths:
        try:
            # compile bytes handles source encodings without writing pyc or importing.
            compile(path.read_bytes(), path.relative_to(root).as_posix(), "exec",
                    flags=ast.PyCF_ONLY_AST, dont_inherit=True)
        except (SyntaxError, UnicodeError, OSError) as exc:
            failures.append({"file": path.relative_to(root).as_posix(),
                             "line": getattr(exc, "lineno", None),
                             "type": type(exc).__name__})
    return {"state": "OK" if paths and not failures else "ERROR",
            "checked": len(paths), "failures": failures}


def command_check(root: Path, logs: Path, name: str, args: list[str]) -> dict:
    log = logs / f"{name}.log"
    try:
        with log.open("wb") as stream:
            result = subprocess.run([sys.executable, *args], cwd=root, stdout=stream,
                                    stderr=subprocess.STDOUT, check=False)
        return {"state": "OK" if result.returncode == 0 else "ERROR",
                "exit_code": result.returncode, "log": file_record(root, log)}
    except OSError:
        return {"state": "ERROR", "exit_code": None, "log": file_record(root, log)}


def junit_breakdown(path: Path) -> dict:
    if not path.is_file():
        return {"state": "MISSING", "groups": {}, "nonpassing": []}
    tree = ET.parse(path)
    groups = defaultdict(Counter)
    nonpassing = []
    for case in tree.iter("testcase"):
        module = case.get("classname", "unknown")
        parts = module.split(".")
        group = ".".join(parts[:2]) if parts[0] == "tests" else parts[0]
        outcome = next((tag for tag in ("error", "failure", "skipped")
                        if case.find(tag) is not None), "passed")
        groups[group][outcome] += 1
        if outcome != "passed":
            nonpassing.append({"module": module, "test": case.get("name", ""),
                               "outcome": outcome})
    return {"state": "OK" if groups else "MISSING",
            "groups": {key: dict(value) for key, value in sorted(groups.items())},
            "nonpassing": nonpassing}


def run_pytest(root: Path, logs: Path, current_commit: str | None) -> dict:
    runner = local_path(root, "scripts/run_tests.py")
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    # This uses the same isolated runner as the canonical test menu.
    result = subprocess.run([sys.executable, str(runner), "--coverage", "--", "-q"],
                            cwd=root, env=env, capture_output=True, check=False)
    (logs / "runner.stdout.log").write_bytes(result.stdout)
    (logs / "runner.stderr.log").write_bytes(result.stderr)
    try:
        summary = json.loads(result.stdout.decode("utf-8"))
        identity = summary["run_id"]
        if not re.fullmatch(r"\d{8}T\d{6}Z_[a-f0-9]{32}", identity):
            raise ValueError("Invalid test identity")
        directory = local_path(root, f"results/tests/{identity}")
        report = test_report(root, directory / "summary.json", current_commit)
        report["breakdown"] = junit_breakdown(directory / "junit.xml")
        report["state"] = ("OK" if result.returncode == 0
                           and report["state"] == "OK"
                           and report["breakdown"]["state"] == "OK"
                           and report["matches_current_clean_commit"] is True
                           else "ERROR" if result.returncode else "WARNING")
        return report
    except (ValueError, OSError, KeyError, TypeError, ET.ParseError):
        return {"state": "ERROR", "exit_code": result.returncode,
                "reason": "No valid summary from this exact runner invocation",
                "log": file_record(root, logs / "runner.stdout.log")}


def print_validation(sections: dict) -> None:
    print("\nRESULTADOS DE VALIDACIÓN DE DESARROLLO")
    for key, value in sections.items():
        print(f"  {key:<16} {value['state']}")
    pytest = sections.get("pytest", {})
    counts = pytest.get("counts") or {}
    if counts:
        passed = (counts["tests"] - counts["errors"] - counts["failures"] - counts["skipped"])
        print(f"  Total: {counts['tests']} | aprobados: {passed} | fallos: {counts['failures']} "
              f"| errores: {counts['errors']} | omitidos: {counts['skipped']}")
        print(f"  Cobertura de líneas: {pytest.get('line_coverage_pct')}%")
    breakdown = pytest.get("breakdown", {})
    for group, values in breakdown.get("groups", {}).items():
        print(f"  {group}: {values}")
    for case in breakdown.get("nonpassing", [])[:30]:
        print(f"    {case['outcome']}: {case['module']}::{case['test']}")
    if len(breakdown.get("nonpassing", [])) > 30:
        print("  Consulte el JSON para los demás casos no aprobados.")


def run_development_validation(root: Path) -> dict:
    root = root.resolve()
    identity = git_identity(root)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    logs = local_path(root, f"logs/sessions/development_{stamp}_{uuid4().hex[:8]}/validation")
    logs.mkdir(parents=True, exist_ok=False)
    sections = {}
    checks = (
        ("syntax", lambda: syntax_check(root, tracked_python(root))),
        ("lint", lambda: command_check(root, logs, "ruff", ["-m", "ruff", "check", "--no-cache", "src", "tests", "scripts"])),
        ("architecture", lambda: command_check(root, logs, "architecture", ["scripts/check_code_size.py", "--check"])),
        ("pytest", lambda: run_pytest(root, logs, identity.get("commit"))),
    )
    for name, check in checks:
        print(f"  Ejecutando {name}...", flush=True)
        try:
            sections[name] = check()
        except Exception as exc:
            sections[name] = {"state": "ERROR", "error_type": type(exc).__name__}
    print_validation(sections)
    return {"state": "OK" if all(v["state"] == "OK" for v in sections.values()) else "WARNING",
            "commit": identity.get("commit"), "checks": sections,
            "logs_directory": logs.relative_to(root).as_posix(),
            "scope": "full local regression; not definitive scientific training"}
