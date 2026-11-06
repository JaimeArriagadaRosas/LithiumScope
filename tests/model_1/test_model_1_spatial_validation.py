import pandas as pd

from lithiumscope.model_1.steps.step_09_validation import (
    build_spatial_groups,
    materialize_regression_splits,
)


def test_model_1_spatial_groups_prevent_neighbor_leakage():
    frame = pd.DataFrame(
        {
            "Longitude": [
                -70.10,
                -70.12,
                -69.10,
                -69.12,
                -68.10,
                -68.12,
            ],
            "Latitude": [
                -23.10,
                -23.12,
                -22.10,
                -22.12,
                -21.10,
                -21.12,
            ],
        }
    )
    groups = build_spatial_groups(
        frame,
        degrees=0.5,
    )
    x = pd.DataFrame(
        {"feature": range(len(frame))}
    )
    y = pd.Series(
        [1.0, 1.2, 2.0, 2.2, 3.0, 3.2]
    )

    splits, strategy = materialize_regression_splits(
        x,
        y,
        groups,
        folds=3,
        seed=42,
    )

    assert strategy == "group_kfold_spatial"
    for train_idx, test_idx in splits:
        train_groups = set(groups.iloc[train_idx])
        test_groups = set(groups.iloc[test_idx])
        assert train_groups.isdisjoint(test_groups)


def test_model_1_validation_falls_back_without_coordinates():
    x = pd.DataFrame(
        {"feature": range(10)}
    )
    y = pd.Series(range(10), dtype=float)

    splits, strategy = materialize_regression_splits(
        x,
        y,
        groups=None,
        folds=5,
        seed=42,
    )

    assert strategy == "kfold_random"
    assert len(splits) == 5
