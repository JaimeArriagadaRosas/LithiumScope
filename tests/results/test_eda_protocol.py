from types import SimpleNamespace
import json

import numpy as np
import pandas as pd
import pytest

from lithiumscope.results.eda_artifacts import complete_run, new_run, verify_eda
from lithiumscope.results.eda_protocol import attach_training_eda, fold_protocol


def cohort():
    x = pd.DataFrame({"a": [1., 2., 3., 4., 5., 6.]})
    y = pd.Series([1., 5., 2., 6., 3., 7.])
    g = pd.Series(["a", "a", "b", "b", "c", "c"])
    splits = [(np.array([2, 3, 4, 5]), np.array([0, 1])),
              (np.array([0, 1, 4, 5]), np.array([2, 3])),
              (np.array([0, 1, 2, 3]), np.array([4, 5]))]
    return x, y, g, splits


def test_eda_identity_is_bound_to_partitions_and_qa():
    x, y, g, splits = cohort()
    base = fold_protocol(x, y, g, splits)
    assert base == fold_protocol(x, y, g, splits)
    assert base != fold_protocol(x, y, g, list(reversed(splits)))
    assert base != fold_protocol(x, y, g, splits, eligible=np.array([1, 0, 1, 1, 1, 1], bool))
    with pytest.raises(ValueError, match="boolean"):
        fold_protocol(x, y, g, splits, eligible=np.ones(len(x), int))


def test_eda_blocks_misalignment_and_monoclass_train():
    x, y, g, splits = cohort()
    with pytest.raises(ValueError, match="identity"):
        fold_protocol(x, y.iloc[::-1], g, splits)
    with pytest.raises(ValueError, match="one class"):
        fold_protocol(x, y, g, splits, endpoints=[100., 100., 100.])


def test_training_eda_reference_requires_hashes_and_exact_protocol(tmp_path):
    x, y, g, splits = cohort()
    protocol = fold_protocol(x, y, g, splits)
    folder = new_run(tmp_path / "eda")
    pd.DataFrame({"fold": [1, 2, 3]}).to_csv(folder / "tables/outer_train_summary.csv", index=False)
    pd.DataFrame({"fold": [1, 2, 3]}).to_csv(folder / "tables/outer_train_features.csv", index=False)
    complete_run(folder, signature_value="s", details={"development_protocol": protocol, "training_reusable": True})
    assert verify_eda(folder, "s")
    assert json.loads((folder / "manifest.json").read_text())["training_reusable"] is False
    context = SimpleNamespace(manifests=tmp_path)
    reference = attach_training_eda(folder, context, protocol)
    assert json.loads(reference.read_text())["fitted_transform_reuse"] is False
    with pytest.raises(ValueError, match="different"):
        attach_training_eda(folder, context, {**protocol, "identity_sha256": "bad"})
    (folder / "tables/outer_train_summary.csv").write_text("changed")
    with pytest.raises(ValueError, match="integrity"):
        attach_training_eda(folder, context, protocol)
