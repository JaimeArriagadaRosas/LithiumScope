import json
from pathlib import Path

from lithiumscope.tools.inspection.development import validation, inventory


def test_syntax_checks_multiple_files_without_importing(tmp_path):
    good = tmp_path / "good.py"
    bad = tmp_path / "bad.py"
    good.write_text("raise RuntimeError('do not execute')\n")
    bad.write_text("def broken(:\n")
    result = validation.syntax_check(tmp_path, [good, bad])
    assert result["state"] == "ERROR" and result["checked"] == 2
    assert result["failures"] == [{"file": "bad.py", "line": 1, "type": "SyntaxError"}]
    assert not list(tmp_path.rglob("*.pyc"))


def test_junit_distinguishes_failures_errors_skips(tmp_path):
    path = tmp_path / "junit.xml"
    path.write_text("""<testsuites><testsuite>
        <testcase classname="tests.model_1.test_fit" name="test_ok"/>
        <testcase classname="tests.model_1.test_fit" name="test_fail"><failure>private</failure></testcase>
        <testcase classname="tests.model_2.test_fit" name="test_error"><error/></testcase>
        <testcase classname="tests.model_2.test_fit" name="test_skip"><skipped/></testcase>
        </testsuite></testsuites>""")
    result = validation.junit_breakdown(path)
    assert result["groups"]["tests.model_1"] == {"passed": 1, "failure": 1}
    assert result["groups"]["tests.model_2"] == {"error": 1, "skipped": 1}
    assert len(result["nonpassing"]) == 3 and "private" not in json.dumps(result)


def test_report_requires_clean_unchanged_commit(tmp_path):
    folder = tmp_path / "results/tests/run"
    folder.mkdir(parents=True)
    summary = folder / "summary.json"
    data = {"state": "passed", "git_commit": "a"*40, "tracked_worktree_dirty": False,
            "git_identity_unchanged": True, "untracked_code_present": False, "counts": {}}
    summary.write_text(json.dumps(data))
    assert inventory.test_report(tmp_path, summary, "a"*40)["matches_current_clean_commit"]
    data["untracked_code_present"] = True
    summary.write_text(json.dumps(data))
    assert not inventory.test_report(tmp_path, summary, "a"*40)["matches_current_clean_commit"]
    data["untracked_code_present"] = False
    data["git_identity_unchanged"] = False
    summary.write_text(json.dumps(data))
    assert not inventory.test_report(tmp_path, summary, "a"*40)["matches_current_clean_commit"]


def test_missing_junit_is_not_passed(tmp_path):
    assert validation.junit_breakdown(Path(tmp_path) / "missing.xml")["state"] == "MISSING"


def test_missing_coverage_cannot_be_reported_as_complete(tmp_path):
    folder = tmp_path / "results/tests/run"
    folder.mkdir(parents=True)
    summary = folder / "summary.json"
    summary.write_text(json.dumps({"state": "passed", "git_commit": "a"*40,
        "tracked_worktree_dirty": False, "git_identity_unchanged": True,
        "untracked_code_present": False, "counts": {"tests": 1, "errors": 0,
        "failures": 0, "skipped": 0}, "exit_code": 0, "coverage_requested": True,
        "coverage_state": "missing_pytest_cov"}))
    result = inventory.test_report(tmp_path, summary, "a"*40)
    assert result["state"] == "WARNING"
    assert result["matches_current_clean_commit"] is True
    assert result["coverage_state"] == "missing_pytest_cov"
    assert result["line_coverage_pct"] is None
    assert result["exit_code"] == 0


def test_runner_detects_missing_coverage_plugin(monkeypatch):
    from scripts import run_tests
    monkeypatch.setattr(run_tests.importlib.util, "find_spec", lambda name: None)
    assert run_tests.coverage_mode(True) == "missing_pytest_cov"
    assert run_tests.coverage_mode(False) == "not_requested"
    monkeypatch.setattr(run_tests.importlib.util, "find_spec", lambda name: object())
    assert run_tests.coverage_mode(True) == "enabled"


def test_junit_canonical_properties_preserve_category_and_markers(tmp_path):
    path = tmp_path / "junit.xml"
    path.write_text("""<testsuite><testcase classname="test_sample" name="test_ok">
        <properties><property name="lithiumscope_test_path" value="tests/unit/data/test_sample.py"/>
        <property name="lithiumscope_test_category" value="unit"/>
        <property name="lithiumscope_test_markers" value="m1,unit"/></properties>
        </testcase></testsuite>""", encoding="utf-8")
    result = validation.junit_breakdown(path)
    assert result["groups"] == {"tests.unit": {"passed": 1}}
    assert result["cases"][0]["path"] == "tests/unit/data/test_sample.py"
    assert result["cases"][0]["markers"] == ["m1", "unit"]
