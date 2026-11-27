"""Contracts for category selection and canonical JUnit identities."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from types import SimpleNamespace

import pytest

from scripts.pytest_classification import CATEGORIES, classify, pytest_collection_modifyitems


def test_classification_uses_canonical_path_and_scope(tmp_path):
    path = tmp_path / "tests" / "unit" / "data" / "test_example.py"
    relative = path.relative_to(tmp_path).as_posix()
    assert classify(tmp_path, path, {relative: ["m1", "eda"]}) == (
        relative, {"unit", "m1", "eda"}
    )
    assert classify(tmp_path, tmp_path / "external.py", {}) is None


def test_noncanonical_tests_fail_explicitly(tmp_path):
    with pytest.raises(ValueError, match="outside canonical layout"):
        classify(tmp_path, tmp_path / "tests" / "old" / "test_example.py", {})


def test_collection_sets_category_and_preserves_explicit_markers(tmp_path, monkeypatch):
    relative = "tests/integration/training/test_example.py"
    monkeypatch.setattr("scripts.pytest_classification.load_scopes",
                        lambda _: {relative: ["m2", "training"]})
    added = []
    item = SimpleNamespace(path=tmp_path / relative, nodeid=relative,
                           iter_markers=lambda: [pytest.mark.slow.mark],
                           add_marker=added.append, user_properties=[])
    pytest_collection_modifyitems(SimpleNamespace(rootpath=tmp_path), [item])
    assert set(added) == {"integration", "m2", "training"}
    props = dict(item.user_properties)
    assert props["lithiumscope_test_path"] == relative
    assert props["lithiumscope_test_category"] == "integration"
    assert set(props["lithiumscope_test_markers"].split(",")) == {
        "integration", "m2", "training", "slow"
    }
    assert len(CATEGORIES) == 8


def test_shared_fixture_is_visible_from_architecture(synthetic_binary_rows):
    assert synthetic_binary_rows["y"] == [0, 0, 1, 1]

@pytest.mark.parametrize("expression, expected", [(None, 2), ("m1 and unit", 1)])
def test_pytest_selection_contract_and_junit_properties(tmp_path, expression, expected):
    root = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
    directory = tmp_path / "tests" / "unit" / "data"
    directory.mkdir(parents=True)
    shutil.copyfile(root / "tests/conftest.py", tmp_path / "tests/conftest.py")
    relative = "tests/unit/data/test_synthetic.py"
    (tmp_path / relative).write_text(
        "def test_m1_case(synthetic_binary_rows):\n"
        "    assert len(synthetic_binary_rows['y']) == 4\n"
        "def test_general_case():\n"
        "    assert True\n", encoding="utf-8")
    scopes = tmp_path / "docs/integral-refactor/test-marker-scopes.json"
    scopes.parent.mkdir(parents=True)
    scopes.write_text(json.dumps({}), encoding="utf-8")
    config = tmp_path / "pytest.ini"
    config.write_text("[pytest]\nmarkers =\n    unit: unit scope\n    m1: M1 scope\n",
                      encoding="utf-8")
    report = tmp_path / "junit.xml"
    command = [sys.executable, "-m", "pytest", "-c", str(config), "-q",
               "--strict-markers", "--junitxml", str(report), "tests"]
    if expression:
        command.extend(["-m", expression])
    env = dict(os.environ, PYTHONPATH=str(root), PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    env.pop("PYTEST_ADDOPTS", None)
    result = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True,
                            text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    cases = list(ET.parse(report).getroot().iter("testcase"))
    assert len(cases) == expected
    for case in cases:
        props = {p.get("name"): p.get("value") for p in case.findall("./properties/property")}
        assert props["lithiumscope_test_path"] == relative
        assert props["lithiumscope_test_category"] == "unit"
        assert ("m1" in props["lithiumscope_test_markers"].split(",")) == (
            case.get("name") == "test_m1_case"
        )
