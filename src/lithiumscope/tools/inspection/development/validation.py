"""Local, temporary validation: syntax, lint, architecture and full pytest evidence."""
from __future__ import annotations

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
    result = subprocess.run(["git", "ls-files", "-z", "main.py", "src", "tests", "scripts"],
                            cwd=root, check=True, capture_output=True, timeout=20)
    paths = []
    for name in result.stdout.decode("utf-8").split("\0"):
        if name.endswith(".py"):
            path = local_path(root, name)
            paths.append(path)
    return sorted(set(paths))


def syntax_check(root: Path, paths: list[Path]) -> dict:
    failures = []
    for path in paths:
        try:
            # compile bytes handles source encodings without writing pyc or importing.
            compile(path.read_bytes(), path.relative_to(root).as_posix(), "exec",
                    dont_inherit=True)
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


def _failure_context(element):
    """Extract only repository-relative code locations and exception class names."""
    raw=element.text or ""
    normalized=raw.replace(chr(92),"/")
    locations=[]
    for match in re.finditer(r"(?<![\w])((?:src/lithiumscope|tests|scripts)/[A-Za-z0-9_./-]+\.py):(\d+)(?::|\b)",normalized):
        path=match.group(1)
        if ".." not in Path(path).parts:
            entry={"file":path,"line":int(match.group(2))}
            if entry not in locations:
                locations.append(entry)
    types=re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*(?:Error|Exception|Failure))\b",element.get("message","")+"\n"+raw)
    return {"locations":locations[-12:],"exception_types":list(dict.fromkeys(types))[-8:],
            "raw_traceback_in_local_junit_only":True}


def junit_breakdown(path: Path) -> dict:
    if not path.is_file():
        return {"state": "MISSING", "groups": {}, "nonpassing": [], "cases": []}
    tree = ET.parse(path)
    groups = defaultdict(Counter)
    nonpassing = []
    cases = []
    for case in tree.iter("testcase"):
        module = case.get("classname", "unknown")
        parts = module.split(".")
        group = ".".join(parts[:2]) if parts[0] == "tests" else parts[0]
        outcome = next((tag for tag in ("error", "failure", "skipped")
                        if case.find(tag) is not None), "passed")
        props = {p.get("name"): p.get("value", "") for p in case.findall("./properties/property")}
        record = {"module": module, "test": case.get("name", ""), "outcome": outcome}
        if "lithiumscope_test_path" in props:
            record["path"] = props["lithiumscope_test_path"]
            record["markers"] = [tag for tag in props.get("lithiumscope_test_markers", "").split(",") if tag]
            category = props.get("lithiumscope_test_category", "")
            if category:
                group = f"tests.{category}"
        groups[group][outcome] += 1
        cases.append(record)
        if outcome != "passed":
            detail=case.find(outcome)
            nonpassing.append({**record,"context":_failure_context(detail)})
    return {"state": "OK" if groups else "MISSING",
            "groups": {key: dict(value) for key, value in sorted(groups.items())},
            "nonpassing": nonpassing, "cases": cases}


def run_pytest(root: Path, logs: Path, current_commit: str | None) -> dict:
    runner = local_path(root, "scripts/run_tests.py")
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env.pop("PYTEST_ADDOPTS", None)
    env["LITHIUMSCOPE_DISABLE_AUTO_INSTALL"] = "1"
    from lithiumscope.core.logging.session import current_session_directory
    env["LITHIUMSCOPE_SESSION_DIR"] = str(current_session_directory())
    # This uses the same isolated runner as the canonical test menu.
    result = subprocess.run([sys.executable, str(runner), "--json", "--coverage", "--", "-q"],
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
        current = git_identity(root)
        trusted = (result.returncode in (0, 1)
                   and result.returncode == summary.get("exit_code")
                   and report["matches_current_clean_commit"] is True
                   and current.get("state") == "OK"
                   and current.get("commit") == current_commit
                   and report["breakdown"]["state"] == "OK")
        report["state"] = ("OK" if result.returncode == 0
                           and report["state"] == "OK"
                           and report["breakdown"]["state"] == "OK"
                           and trusted
                           else "ERROR" if result.returncode else "WARNING")
        from lithiumscope.tools.inspection.development.criteria import evaluate_catalog, load_catalog
        report["criterion_evidence"] = evaluate_catalog(
            load_catalog(root), report["breakdown"].pop("cases"),
            trusted_run=trusted,
        )
        return report
    except (ValueError, OSError, KeyError, TypeError, ET.ParseError):
        return {"state": "ERROR", "exit_code": result.returncode,
                "reason": "No valid summary from this exact runner invocation",
                "log": file_record(root, logs / "runner.stdout.log")}


def print_validation(sections: dict) -> None:
    print("\nRESULTADOS DE VALIDACIÓN DE DESARROLLO")
    for key, value in sections.items():
        suffix = " (evidencia histórica opcional)" if value.get("optional_evidence") else ""
        print(f"  {key:<16} {value['state']}{suffix}")
    pytest = sections.get("pytest", {})
    if pytest.get("coverage_state") == "missing_pytest_cov":
        print("  [PENDIENTE] pytest-cov ausente; regresión ejecutada sin cobertura.")
    if pytest.get("exit_code") is not None:
        print(f"  Código de salida pytest: {pytest['exit_code']}")
    counts = pytest.get("counts") or {}
    if all(isinstance(counts.get(key), int) for key in ("tests", "errors", "failures", "skipped")):
        passed = (counts["tests"] - counts["errors"] - counts["failures"] - counts["skipped"])
        print(f"  Total: {counts['tests']} | aprobados: {passed} | fallos: {counts['failures']} "
              f"| errores: {counts['errors']} | omitidos: {counts['skipped']}")
        print(f"  Cobertura de líneas: {pytest.get('line_coverage_pct')}%")
    for state, total in pytest.get("criterion_evidence", {}).get("summary", {}).items():
        print(f"  Criterios asociados: {state}: {total}")
    breakdown = pytest.get("breakdown", {})
    for group, values in breakdown.get("groups", {}).items():
        print(f"  {group}: {values}")
    for case in breakdown.get("nonpassing", [])[:30]:
        print(f"    {case['outcome']}: {case['module']}::{case['test']}")
    if len(breakdown.get("nonpassing", [])) > 30:
        print("  Consulte el JSON para los demás casos no aprobados.")


def _check_dependencies(root):
    from lithiumscope.tools.inspection.dependencies import audit_imports
    report=audit_imports(root)
    blockers=("syntax_errors","eager_cycles","cross_model_edges","core_to_model_edges",
              "reporting_to_science_edges","unresolved_local_imports","retired_external_references")
    return {"state":"ERROR" if any(report[k] for k in blockers) else "OK",**report}


def _benchmark_check(root, logs, *, execute):
    if not execute:
        return {"state":"SKIPPED","reason":"Synthetic benchmark not authorized"}
    from lithiumscope.resources.monitoring.acceleration_benchmark import compare
    output=logs/"acceleration_benchmark"
    report=compare(output,execute=True)
    complete=report.get("state")=="complete" and report.get("cpu_equivalent") is True
    return {"state":"OK" if complete and report.get("parallel_state")=="measured" else
                    "WARNING" if complete else "ERROR",
            "scope":report["scope"],"cpu_equivalent":report.get("cpu_equivalent"),
            "parallel_state":report.get("parallel_state"),"gpu_state":report.get("gpu_state"),
            "observed_speedup":report.get("observed_speedup"),
            "runs":report.get("runs",[]),"evidence":file_record(root,output/"comparison.json")}


def _historical_parity_check(root,logs):
    from lithiumscope.tools.inspection.development.historical_parity import verify
    report = verify(root,logs/"historical_parity.json")
    # Absence of an archived model is not a failure of the next training pipeline.
    if report.get("state") == "BLOCKED" and report.get("reason") in (
            "historical_artifacts_absent", "historical_imagery_absent"):
        report = {**report, "required_for_new_training": False, "optional_evidence": True}
    return report


def run_development_validation(root: Path, *, include_benchmark=False) -> dict:
    root = root.resolve()
    identity = git_identity(root)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    from lithiumscope.core.logging.session import current_session_directory
    logs = current_session_directory() / "validation" / f"development_{stamp}_{uuid4().hex[:8]}"
    logs.mkdir(parents=True, exist_ok=False)
    sections = {}
    from lithiumscope.tools.inspection.development.protected import protected_inventory, compare_protected
    before = protected_inventory(root)
    checks = (
        ("syntax", lambda: syntax_check(root, tracked_python(root))),
        ("lint", lambda: command_check(root, logs, "ruff", ["-m", "ruff", "check", "--no-cache", "main.py", "src", "tests", "scripts"])),
        ("architecture", lambda: command_check(root, logs, "architecture", ["scripts/check_code_size.py", "--check"])),
        ("imports", lambda: _check_dependencies(root)),
        ("pytest", lambda: run_pytest(root, logs, identity.get("commit"))),
        ("historical_parity", lambda: _historical_parity_check(root,logs)),
        *((("benchmark", lambda: _benchmark_check(root,logs,execute=True)),) if include_benchmark else ()),
    )
    for name, check in checks:
        print(f"  Ejecutando {name}...", flush=True)
        try:
            sections[name] = check()
        except Exception as exc:
            from lithiumscope.core.logging.session import get_logger
            get_logger("development.validation").exception("Development check failed: %s",name)
            sections[name] = {"state": "ERROR", "error_type": type(exc).__name__}
    sections["protected_artifacts"] = compare_protected(before, protected_inventory(root))
    if sections["protected_artifacts"]["state"] != "OK":
        for name in sections["protected_artifacts"]["changed_paths"]:
            print(f"  [ERROR] Archivo científico modificado: {name}")
    print_validation(sections)
    required = [v for k,v in sections.items() if not (k == "historical_parity" and v.get("optional_evidence") is True)]
    return {"state": "OK" if all(v["state"] == "OK" for v in required) else "WARNING",
            "commit": identity.get("commit"), "checks": sections,
            "logs_directory": logs.relative_to(root).as_posix(),
            "scope": "full local regression; not definitive scientific training"}
