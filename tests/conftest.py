"""Reusable synthetic fixtures are declared locally; never production evidence."""
import pytest

from scripts.pytest_classification import pytest_collection_modifyitems  # noqa: F401

@pytest.fixture
def synthetic_binary_rows():
    return {"x":[[0.],[1.],[2.],[3.]],"y":[0,0,1,1]}
