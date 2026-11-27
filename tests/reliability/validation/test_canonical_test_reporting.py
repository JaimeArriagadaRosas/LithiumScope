"""Evidence reporting: category, overlapping scopes and full-suite reconciliation."""
import json

from scripts.pytest_console import group_of
from scripts.software_checks import validation_outcome
from scripts.test_reporting.architecture import audit, CATEGORIES, SCOPES
from scripts.test_reporting.evidence import collection_evidence
from scripts.test_reporting.outcomes import report_details


def test_static_architecture_accepts_complete_catalog_then_detects_duplicate(tmp_path):
    modules = []
    for category in CATEGORIES:
        path = tmp_path / "tests" / category / "contracts" / f"test_{category}.py"
        path.parent.mkdir(parents=True)
        path.write_text("def test_example():\n    assert True\n", encoding="utf-8")
        modules.append(path.relative_to(tmp_path).as_posix())
    (tmp_path / "tests/conftest.py").write_text("", encoding="utf-8")
    docs = tmp_path / "docs/integral-refactor"
    docs.mkdir(parents=True)
    (docs / "test-relocation-purpose.json").write_text(json.dumps({
        "tests/old/test_architecture.py": modules[0]
    }))
    ids = [f"M1-{i:02}" for i in range(1, 28)] + [f"M2-{i:02}" for i in range(1, 31)]
    ids += [f"ARQ-{i:03}" for i in range(1, 110)]
    (docs / "criterion-tests.json").write_text(json.dumps({
        "schema_version": 1, "criteria": {key: [modules[0]] for key in ids}
    }))
    (docs / "test-marker-scopes.json").write_text(json.dumps({modules[0]: ["m1"]}))
    (tmp_path / "pyproject.toml").write_text(
        '[tool.pytest.ini_options]\ntestpaths = ["tests"]\nmarkers = '
        + json.dumps(list(CATEGORIES + SCOPES)))
    report = audit(tmp_path)
    assert report["state"] == "OK", report
    assert report["modules"] == 8 and report["criterion_references"] == 166
    old = tmp_path / "tests/old/test_architecture.py"
    old.parent.mkdir()
    old.write_text("pass")
    report = audit(tmp_path)
    assert report["state"] == "ERROR"
    assert any("Duplicate basename" in issue for issue in report["problems"])


def _report(tmp_path):
    path = "tests/unit/data/test_example.py"
    module = tmp_path / path
    module.parent.mkdir(parents=True)
    module.write_text("pass")
    junit = tmp_path / "junit.xml"
    junit.write_text(f"""<testsuite>
      <testcase classname="test_example" name="test_ok" time="1">
        <properties>
          <property name="lithiumscope_test_path" value="{path}"/>
          <property name="lithiumscope_test_category" value="unit"/>
          <property name="lithiumscope_test_markers" value="unit,m1,eda"/>
        </properties>
      </testcase>
    </testsuite>""")
    return path, report_details(junit, tmp_path / "absent.xml", root=tmp_path)


def test_report_uses_canonical_path_and_overlapping_scopes(tmp_path):
    path, report = _report(tmp_path)
    assert report["categories"]["unit"] == {"passed": 1}
    assert report["responsibilities"]["unit/data"] == {"passed": 1}
    assert report["scopes"]["m1"] == report["scopes"]["eda"] == {"passed": 1}
    assert report["scope_counts_overlap"] is True
    assert report["cases"][0]["path"] == path
    assert report["line_coverage_pct"] is None
    assert group_of(path + "::test_ok") == "unit/data"


def test_complete_collection_requires_reconciled_cases_and_no_deselection(tmp_path):
    path, report = _report(tmp_path)
    progress = tmp_path / "progress.jsonl"
    events = [{"kind": "collected", "total": 1, "module_counts": {path: 1}},
              {"kind": "session_finished", "exit_code": 0}]
    progress.write_text("\n".join(map(json.dumps, events)))
    def evidence():
        return collection_evidence(tmp_path, progress, report["cases"],
                                   junit_state=report["junit_state"], require_full=True)
    assert evidence()["full_suite_verified"]
    events.append({"kind": "deselected", "total": 1})
    progress.write_text("\n".join(map(json.dumps, events)))
    incomplete = evidence()
    assert not incomplete["full_suite_verified"]
    assert validation_outcome(0, {}, {"executed_tests": 1, "collection": incomplete},
        coverage_requested=False, clean=True, counts={"tests": 1}) == ("incomplete", 10)


def test_explicit_invalid_identity_never_falls_back_to_basename(tmp_path):
    path, _ = _report(tmp_path)
    junit = tmp_path / "junit.xml"
    junit.write_text(junit.read_text().replace('value="unit"/>', 'value="scientific"/>'))
    report = report_details(junit, tmp_path / "absent.xml", root=tmp_path)
    assert report["cases"][0]["path"] == ""
    assert report["unclassified"] == {"passed": 1}
    assert not report["cases"][0]["canonical_identity"]
