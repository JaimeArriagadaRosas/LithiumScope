import json
import sys

from scripts import software_checks
from scripts.pytest_console import report_details


def test_full_validation_cannot_pass_with_failed_preflight_or_missing_coverage():
    details = {"executed_tests": 5, "line_coverage_pct": 80.}
    counts = {"tests": 5, "skipped": 0}
    args = {"coverage_requested": True, "clean": True, "counts": counts}
    assert software_checks.validation_outcome(0, {"imports": {"state": "ERROR"}}, details, **args) == ("failed", 1)
    assert software_checks.validation_outcome(0, {"imports": {"state": "OK"}}, details, **args) == ("passed", 0)
    assert software_checks.validation_outcome(0, {}, {**details, "line_coverage_pct": None}, **args) == ("incomplete", 10)
    assert software_checks.validation_outcome(2, {}, details, **args) == ("failed", 2)
    assert software_checks.validation_outcome(0, {}, details, **{**args, "counts": {"tests": 5, "skipped": 1}}) == ("incomplete", 10)


def test_collection_errors_are_grouped_by_real_module_and_cause(tmp_path):
    junit = tmp_path / "junit.xml"
    junit.write_text("""<testsuite>
        <testcase classname="" name="tests.model_2.test_source" time="0"><error message="collection failure">E ModuleNotFoundError: No module named 'missing.module'</error></testcase>
        <testcase classname="tests.core.test_ok" name="test_ok" time="0.1"/>
        </testsuite>""")
    details = report_details(junit, tmp_path / "missing.xml")
    assert details["collection_errors"] == 1
    assert details["executed_tests"] == 1
    assert details["groups"]["tests.model_2"]["error"] == 1
    assert details["collection_causes"] == {"ModuleNotFoundError: missing.module": 1}
    assert details["slowest"][0]["test"] == "tests.core.test_ok::test_ok"


def test_preflight_executes_script_and_keeps_independent_checks(tmp_path, monkeypatch):
    calls = []
    def fake(root, logs, name, arguments, env, **kwargs):
        calls.append((name, arguments))
        return {"state": "ERROR" if name == "retired_imports" else "OK"}
    monkeypatch.setattr(software_checks, "_command", fake)
    results = software_checks.run_preflight(tmp_path, tmp_path, {}, show=False)
    assert calls[0] == ("retired_imports", ["scripts/check_internal_imports.py"])
    assert len(results) == 6 and results["startup_imports"]["state"] == "OK"


def test_protected_artifact_change_is_not_certified():
    before = {"state": "OK", "inventory": {"model": {"state": "OK", "sha256": "a"}}}
    after = {"state": "OK", "inventory": {"model": {"state": "OK", "sha256": "b"}}}
    assert software_checks.compare_artifacts(before, after)["changed_paths"] == ["model"]
