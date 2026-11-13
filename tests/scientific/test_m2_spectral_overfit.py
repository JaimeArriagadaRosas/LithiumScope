"""Regression tests: spectral augmentation and leakage-safe M2 ablations."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lithiumscope.model_2.methodology.spectral_augmentation import (
    augmented_feature_record, perturb_sentinel_dn, seed_for_sample,
)
from lithiumscope.model_2.methodology.spectral_overfit_pilot import (
    embargo_training_indices, train_and_score,
)


def test_augmentation_is_repeatable_and_preserves_band_masks():
    cube = np.arange(6 * 8 * 8, dtype=np.float32).reshape(6, 8, 8) + 1000
    cube[:, 0, 0] = np.nan
    original = cube.copy()
    first = perturb_sentinel_dn(cube, seed=seed_for_sample("A-17"))
    second = perturb_sentinel_dn(cube, seed=seed_for_sample("A-17"))
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(cube, original)
    assert np.array_equal(np.isfinite(first), np.isfinite(cube))
    assert np.all((first[np.isfinite(first)] / cube[np.isfinite(cube)]) < 1.04)
    assert np.all((first[np.isfinite(first)] / cube[np.isfinite(cube)]) > 0.96)


def test_out_of_schema_or_excessive_jitter_is_rejected():
    with pytest.raises(ValueError, match="six-band"):
        perturb_sentinel_dn(np.ones((3, 8, 8)), seed=3)
    with pytest.raises(ValueError, match="Unsafe"):
        perturb_sentinel_dn(np.ones((6, 8, 8)), seed=3, common_gain_amplitude=0.25)


def test_spectral_schema_has_126_features_only():
    cube = np.ones((6, 8, 8), dtype=np.float32) * 2000
    record = augmented_feature_record(cube, sample_id="A")
    assert len(record) == 126
    assert not any("li_" in key.lower() or "latitude" in key for key in record)
    assert np.isfinite(list(record.values())).all()


def test_true_embargo_removes_nearby_training_before_fit():
    coords = np.array([
        [-23.0, -70.0],
        [-23.002, -70.002],
        [-24.0, -71.0],
        [-23.001, -70.001],
    ])
    remaining = embargo_training_indices(
        np.array([0, 1, 2]), np.array([3]), coords, km=5.0
    )
    np.testing.assert_array_equal(remaining, [2])


def test_augmented_rows_are_fit_only_and_weighted(monkeypatch):
    state = {}

    class SpyModel:
        def fit(self, x, y, sample_weight=None):
            state["train_features"] = x.copy()
            state["train_labels"] = np.asarray(y).copy()
            state["weights"] = np.asarray(sample_weight)
            return self

        def predict_proba(self, x):
            # Values are a deterministic function of supplied features.
            p = np.clip(np.asarray(x.iloc[:, 0], dtype=float) / 100, 0.01, 0.99)
            return np.column_stack([1 - p, p])

    monkeypatch.setattr(
        "lithiumscope.model_2.methodology.spectral_overfit_pilot.create_model",
        lambda **_kwargs: SpyModel(),
    )
    columns = [f"feature_{i}" for i in range(126)]
    data = pd.DataFrame(np.arange(4 * 126).reshape(4, 126) / 100, columns=columns)
    augmented = data + 0.25
    labels = np.array([0, 1, 0])
    original_scores, apparent, chosen = train_and_score(
        data, augmented, labels, np.array([0, 1, 2]), np.array([3]),
        strategy="all126", augment_train=True, seed=42,
    )
    assert len(chosen) == 126
    assert len(state["train_features"]) == 6
    np.testing.assert_array_equal(state["train_labels"], [0, 1, 0, 0, 1, 0])
    np.testing.assert_allclose(state["weights"], [0.5] * 6)
    np.testing.assert_allclose(original_scores, np.array([data.iloc[3, 0] / 100]))
    assert len(apparent) == 3
