"""Contracts for category selection and canonical JUnit identities."""
from pathlib import Path
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
