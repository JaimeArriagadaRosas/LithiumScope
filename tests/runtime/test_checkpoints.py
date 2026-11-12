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
