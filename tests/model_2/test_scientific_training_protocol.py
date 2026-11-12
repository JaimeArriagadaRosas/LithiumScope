import numpy as np
import pandas as pd
import pytest

from lithiumscope.model_2.training.scientific_validation import (
    make_scientific_folds, scientific_labels, evaluate_scientific_baseline,
)


def _example():
    # Different Li distributions in the groups to catch global target leakage.
    groups = pd.Series(np.repeat(np.arange(12), 6))
    lithium = pd.Series(
        np.concatenate([np.arange(6) + i * 3.0 for i in range(12)]),
        dtype=float,
    )
    return lithium, groups


def test_scientific_folds_have_no_overlap_and_fold_local_q75():
    lithium, groups = _example()
    folds = make_scientific_folds(lithium, groups, n_splits=3)
    for fold in folds:
        assert set(groups.iloc[fold.train_idx]).isdisjoint(
            set(groups.iloc[fold.test_idx])
        )
        expected = np.quantile(lithium.iloc[fold.train_idx], .75)
        assert fold.label_threshold_ppm == expected
        np.testing.assert_array_equal(
            fold.y_test, lithium.iloc[fold.test_idx].to_numpy() >= expected,
        )
    values = scientific_labels(folds, len(lithium))
    assert len(values) == len(lithium)
    assert set(np.unique(values)) == {0, 1}


def test_scientific_folds_reject_missing_groups():
    lithium, _ = _example()
    with pytest.raises(ValueError, match="grupos espaciales"):
        make_scientific_folds(lithium, None)


def test_folds_and_baseline_share_labels_and_groups():
    lithium, groups = _example()
    folds = make_scientific_folds(lithium, groups, n_splits=3)
    row, table, scores = evaluate_scientific_baseline(folds, len(lithium))
    assert row["is_baseline"] is True
    assert table["group_overlap"].eq(0).all()
    for fold in folds:
        np.testing.assert_allclose(
            scores[fold.test_idx], np.mean(fold.y_train),
        )


def test_global_shift_does_not_change_fold_local_labels():
    lithium, groups = _example()
    a = make_scientific_folds(lithium, groups, n_splits=3)
    b = make_scientific_folds(lithium + 100, groups, n_splits=3)
    for left, right in zip(a, b):
        np.testing.assert_array_equal(left.y_test, right.y_test)
        assert right.label_threshold_ppm == pytest.approx(
            left.label_threshold_ppm + 100,
        )
