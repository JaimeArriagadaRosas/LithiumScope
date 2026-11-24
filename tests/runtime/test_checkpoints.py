import numpy as np

from lithiumscope.core.checkpoints import FoldCheckpointStore


def test_fold_checkpoint_roundtrip(tmp_path):
    store = FoldCheckpointStore(tmp_path, "random_forest")
    store.save_fold(
        1,
        [0, 2],
        [1.5, 2.5],
        {"rmse": 1.0, "mae": 0.8, "r2": 0.5},
        params={"n_estimators": 100},
    )

    payload = store.load_fold(1, [0, 2])

    assert payload is not None
    assert np.allclose(payload["predictions"], [1.5, 2.5])
    assert payload["params"]["n_estimators"] == 100


def test_fold_checkpoint_rejects_changed_split(tmp_path):
    store = FoldCheckpointStore(tmp_path, "random_forest")
    store.save_fold(
        1,
        [0, 2],
        [1.5, 2.5],
        {"rmse": 1.0},
    )

    assert store.load_fold(1, [0, 3]) is None


def test_checkpoint_detects_modified_payload(tmp_path):
    import json
    store = FoldCheckpointStore(tmp_path, "rf")
    path = store.save_fold(1, [0, 2], [1.5, 2.5], {"rmse": 1.0})
    envelope = json.loads(path.read_text(encoding="utf-8"))
    envelope["payload"]["predictions"][0] = 100
    path.write_text(json.dumps(envelope), encoding="utf-8")
    assert store.load_fold(1, [0, 2]) is None


def test_checkpoint_rejects_nonfinite_and_duplicate_indices(tmp_path):
    import pytest
    store = FoldCheckpointStore(tmp_path, "rf")
    with pytest.raises(ValueError):
        store.save_fold(1, [0, 2], [1.5, np.nan], {})
    with pytest.raises(ValueError):
        store.save_fold(1, [0, 0], [1.5, 2.5], {})


def test_algorithm_checkpoint_detects_modification(tmp_path):
    import json
    store = FoldCheckpointStore(tmp_path, "rf")
    path = store.save_algorithm({"params": {"depth": 3}})
    assert store.load_algorithm()["params"]["depth"] == 3
    data = json.loads(path.read_text(encoding="utf-8"))
    data["payload"]["params"]["depth"] = 99
    path.write_text(json.dumps(data), encoding="utf-8")
    assert store.load_algorithm() is None
