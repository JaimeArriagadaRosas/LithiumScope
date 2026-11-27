"""Historical class paths resolve only during trusted M1 deserialization."""
import gzip
from io import BytesIO
from pathlib import Path
import sys

import joblib
import numpy as np
import pytest

from lithiumscope.model_1.training.preprocessing.tabular import FeatureSchema
from lithiumscope.model_1.training.algorithms.tabnet import TabNetRegressorAdapter
from lithiumscope.persistence.model_deserialization import load_model_artifact


@pytest.mark.parametrize("module,name,canonical", [
    ("lithiumscope.model_1.steps.step_08_preprocessing", "FeatureSchema", FeatureSchema),
    ("lithiumscope.model_1.training.tabnet", "TabNetRegressorAdapter", TabNetRegressorAdapter),
])
@pytest.mark.parametrize("compressed", [False, True])
def test_historical_class_reference_without_import_alias(module, name, canonical, compressed):
    payload = f"c{module}\n{name}\n.".encode("ascii")
    if compressed:
        payload = gzip.compress(payload)
    assert load_model_artifact(BytesIO(payload), model_group="model_1") is canonical
    assert module not in sys.modules


@pytest.mark.parametrize("compression", [0, 3])
def test_canonical_bundle_preserves_schema_and_arrays(tmp_path, compression):
    path = tmp_path / "model.joblib"
    schema = FeatureSchema(["SiO2"], [])
    array = np.arange(12, dtype=float).reshape(3, 4)
    joblib.dump({"schema": schema, "array": array}, path, compress=compression)
    restored = load_model_artifact(path, model_group="model_1")
    assert restored["schema"] == schema
    np.testing.assert_array_equal(restored["array"], array)


def test_retired_source_modules_do_not_exist():
    root = Path(__file__).resolve().parents[2] / "src/lithiumscope/model_1"
    assert not list((root / "steps").rglob("*.py"))
    assert not (root / "training/tabnet.py").exists()
