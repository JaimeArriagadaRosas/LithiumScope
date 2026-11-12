"""Frozen M2 source/fold and development comparison safety contracts."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from lithiumscope.model_2.evaluation.canonical_protocol import validate_folds
from lithiumscope.model_2.methodology.canonical_rf_comparison import _metrics


def _fixture():
    # Test only the generic isolation invariant; real 783-row contract is
    # audited separately on the actual frozen sources before any training.
    manifest = pd.DataFrame({
        "sample_id": ["s0", "s1", "s2", "s3"],
        "spatial_group": ["a", "b", "c", "d"],
        "Li_icpms": [4.0, 28.0, 5.0, 31.0],
    })
    folds = [
        SimpleNamespace(
            train_idx=np.array([0, 1]), test_idx=np.array([2, 3]),
            y_train=np.array([0, 1]), y_test=np.array([0, 1]),
            label_threshold_ppm=21.6,
        ),
        SimpleNamespace(
            train_idx=np.array([2, 3]), test_idx=np.array([0, 1]),
            y_train=np.array([0, 1]), y_test=np.array([0, 1]),
            label_threshold_ppm=21.6,
        ),
    ]
    return manifest, folds


def test_m2_canonical_metrics_are_classification_not_artificial_holdout():
    y = np.array([0, 1, 0, 1])
    p = np.array([0.1, 0.6, 0.4, 0.8])
    got = _metrics(y, p, cutoff=0.5)
    assert got["roc_auc"] == 1.0
    assert got["balanced_accuracy"] == 1.0
    assert got["recall"] == 1.0
    assert got["predicted_positives"] == 2


def test_canonical_folds_reject_inadequate_source_population():
    manifest, folds = _fixture()
    with pytest.raises(ValueError, match="canonical source"):
        validate_folds(manifest, np.array([True] * 4), folds)


def test_canonical_metrics_reject_single_class_holdout():
    with pytest.raises(ValueError, match="Invalid outer"):
        _metrics([0, 0, 0], [0.2, 0.3, 0.1], cutoff=0.5)


def test_frozen_data_snapshot_has_no_release_approval_fields():
    from lithiumscope.model_2.evaluation.canonical_protocol import INPUTS
    assert set(INPUTS) >= {
        "manifest", "features", "quality", "frozen_folds",
        "reference_model", "active_registry",
    }
    from lithiumscope.model_2.methodology.canonical_rf_comparison import PROFILES
    assert PROFILES == ("legacy", "regularized_v2")


def _valid_full_size_fixture():
    from sklearn.model_selection import GroupKFold

    n = 783
    labels = np.arange(n) % 2
    source = pd.DataFrame({
        "sample_id": [f"id_{i}" for i in range(n)],
        "spatial_group": [f"region_{i // 6}" for i in range(n)],
        "Li_icpms": np.where(labels, 29.0, 11.0),
    })
    eligible = np.ones(n, dtype=bool)
    eligible[-16:] = False
    folds = []
    for train, test in GroupKFold(5).split(
        source, groups=source.spatial_group
    ):
        folds.append(SimpleNamespace(
            train_idx=train, test_idx=test,
            y_train=labels[train], y_test=labels[test],
            label_threshold_ppm=21.6,
        ))
    return source, eligible, folds


def test_canonical_group_fold_contract_preserves_every_outer_case():
    source, eligible, folds = _valid_full_size_fixture()
    info = validate_folds(source, eligible, folds)
    assert len(info) == 5
    assert sum(row["test_rows_unfiltered"] for row in info) == 783
    assert all(row["train_rows_after_qa95"] < 783 for row in info)
    assert all(row["test_positives"] > 0 for row in info)


def test_canonical_folds_reject_overlap_even_with_valid_row_counts():
    source, eligible, folds = _valid_full_size_fixture()
    changed = list(folds)
    first = changed[0]
    changed[0] = SimpleNamespace(
        train_idx=np.concatenate([first.train_idx, first.test_idx[:1]]),
        test_idx=first.test_idx,
        y_train=np.concatenate([first.y_train, first.y_test[:1]]),
        y_test=first.y_test, label_threshold_ppm=21.6,
    )
    with pytest.raises(ValueError, match="overlap"):
        validate_folds(source, eligible, changed)


def test_geographic_proximity_audit_is_measured_in_km():
    from lithiumscope.model_2.evaluation.spatial_distance_sensitivity import (
        distances_km,
    )

    nearby = distances_km(
        np.array([[-33.45, -70.65], [-33.45, -70.66]]),
        np.array([[-33.45, -70.65]]),
    )
    assert nearby.shape == (2, 1)
    assert abs(nearby[0, 0]) < 1e-10
    assert 0.5 < nearby[1, 0] < 1.5
    with pytest.raises(ValueError, match="Lat/lon"):
        distances_km(np.array([[42.0]]), np.array([[42.0]]))


def test_outer_test_labels_cannot_change_training_or_predictions(
    monkeypatch, tmp_path,
):
    import lithiumscope.model_2.methodology.canonical_rf_comparison as study

    rng = np.random.default_rng(42)
    size = 60
    group = np.repeat([f"site_{i}" for i in range(6)], 10)
    labels = np.arange(size) % 2
    lithium = np.where(labels, 28.0, 14.0)
    features = pd.DataFrame(
        rng.normal(size=(size, 6)),
        columns=[f"band_{i}" for i in range(6)],
    )
    features.iloc[:, 0] += labels * 0.4
    manifest = pd.DataFrame({
        "sample_id": [f"row_{i}" for i in range(size)],
        "spatial_group": group,
        "Li_icpms": lithium,
    })
    train = np.arange(40)
    test = np.arange(40, 60)
    fold = SimpleNamespace(
        train_idx=train, test_idx=test,
        y_test=labels[test], label_threshold_ppm=21.6,
    )
    qa = np.ones(size, dtype=bool)
    qa[2] = False
    calls = []

    def internal_optimizer(*args, **kwargs):
        x_train, y_train = args[2:4]
        calls.append((len(x_train), len(y_train)))
        return {
            "n_estimators": 12,
            "max_depth": 3,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
        }

    monkeypatch.setattr(
        study, "optimize_classifier_with_optuna", internal_optimizer
    )
    first = study._execute_arm(
        manifest=manifest, features=features, fold=fold, eligible=qa,
        fold_number=1, profile="regularized_v2", budget=2,
        output=tmp_path / "first", protocol_sha="frozen-synthetic",
    )
    altered = manifest.copy()
    altered.loc[test, "Li_icpms"] = 999.0
    second = study._execute_arm(
        manifest=altered, features=features, fold=fold, eligible=qa,
        fold_number=1, profile="regularized_v2", budget=2,
        output=tmp_path / "second", protocol_sha="frozen-synthetic",
    )
    assert calls == [(39, 39), (39, 39)]
    assert first["test_unfiltered"] == 20
    assert first["train_eligible"] == 39
    assert first["evaluations"]["raw"]["scores"] == second["evaluations"]["raw"]["scores"]
    assert np.allclose(
        first["evaluations"]["inner_sigmoid"]["scores"],
        second["evaluations"]["inner_sigmoid"]["scores"],
        rtol=0.0, atol=1e-12,
    )
