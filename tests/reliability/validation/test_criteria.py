from pathlib import Path

from lithiumscope.tools.inspection.development.criteria import evaluate_catalog, expected_ids, load_catalog


def test_catalog_associates_all_166_criteria():
    root = next(parent for parent in Path(__file__).resolve().parents if (parent / "pyproject.toml").is_file())
    assert set(load_catalog(root)) == expected_ids()
    assert len(expected_ids()) == 166


def test_associated_pass_does_not_close_criterion():
    criteria = {"M1-01": ["tests/model_1/test_contract.py"]}
    cases = [{"module": "tests.model_1.test_contract.TestContract", "outcome": "passed"}]
    result = evaluate_catalog(criteria, cases, trusted_run=True)["criteria"]["M1-01"]
    assert result["state"] == "ASSOCIATED_TESTS_PASSED"
    assert result["criterion_closed"] is False


def test_failed_skipped_missing_and_untrusted_are_not_passed():
    criteria = {"M1-01": ["tests/test_contract.py", "tests/test_other.py"]}
    cases = [{"module": "test_contract", "outcome": "failure"}]
    assert evaluate_catalog(criteria, cases, trusted_run=True)["criteria"]["M1-01"]["state"] == "FAILED_TEST"
    cases[0]["outcome"] = "skipped"
    assert evaluate_catalog(criteria, cases, trusted_run=True)["criteria"]["M1-01"]["state"] == "INCOMPLETE_TEST_EVIDENCE"
    assert evaluate_catalog(criteria, [], trusted_run=False)["criteria"]["M1-01"]["state"] == "UNVERIFIED_RUN"


def test_catalog_allows_more_than_three_relevant_modules():
    root=next(parent for parent in Path(__file__).resolve().parents if (parent / "pyproject.toml").is_file())
    catalog=load_catalog(root)
    assert any(len(modules)>3 for modules in catalog.values())


def test_duplicate_stems_require_qualified_evidence():
    criteria={"M1-01":["tests/model_1/test_same.py"],"M2-01":["tests/model_2/test_same.py"]}
    result=evaluate_catalog(criteria,[{"module":"test_same","outcome":"passed"}],trusted_run=True)["criteria"]
    assert all(row["state"]=="INCOMPLETE_TEST_EVIDENCE" for row in result.values())
    result=evaluate_catalog(criteria,[{"module":"tests.model_1.test_same.TestClass","outcome":"passed"}],trusted_run=True)["criteria"]
    assert result["M1-01"]["state"]=="ASSOCIATED_TESTS_PASSED"
    assert result["M2-01"]["state"]=="INCOMPLETE_TEST_EVIDENCE"


def test_explicit_canonical_path_overrides_junit_classname():
    criteria = {"M1-01": ["tests/unit/data/test_same.py"], "M2-01": ["tests/scientific/features/test_same.py"]}
    cases = [{"path": "tests/scientific/features/test_same.py", "module": "tests.unit.data.test_same",
              "outcome": "passed"}]
    result = evaluate_catalog(criteria, cases, trusted_run=True)["criteria"]
    assert result["M1-01"]["state"] == "INCOMPLETE_TEST_EVIDENCE"
    assert result["M2-01"]["state"] == "ASSOCIATED_TESTS_PASSED"
    cases[0]["path"] = "tests/unknown/test_same.py"
    result = evaluate_catalog(criteria, cases, trusted_run=True)["criteria"]
    assert all(row["state"] == "INCOMPLETE_TEST_EVIDENCE" for row in result.values())
