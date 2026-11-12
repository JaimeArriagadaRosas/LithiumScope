"""Scientific M2 source independence, spectral selection and release gates."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lithiumscope.model_2.methodology.independence_gate import (
    audit_external_overlap,
    pairwise_distances_km,
    release_gate,
)
from lithiumscope.model_2.methodology.spectral_feature_policy import (
    BANDS, STATS, select_spectral_features,
)
from lithiumscope.experiments.analysis.m2_generalization_pilot import (
    metrics, model_params, ARMS,
)


def test_physically_predeclared_48_features_only():
    direct = [f"{band}_{stat}" for band in BANDS for stat in STATS]
    full = direct + [f"derived_index_{i}" for i in range(78)]
    assert len(full) == 126
    assert select_spectral_features(full, "full_126") == full
    assert select_spectral_features(full, "direct_band_48") == direct
    assert "derived_index_0" not in select_spectral_features(
        full, "direct_band_48"
    )
    with pytest.raises(ValueError, match="missing"):
        select_spectral_features(full[:-80], "direct_band_48")
    with pytest.raises(ValueError, match="Unsupported"):
        select_spectral_features(full, "test_tuned_from_external")


def test_training_only_regularization_is_stricter():
    class Model:
        def get_params(self):
            return dict(max_depth=16, min_samples_leaf=3, n_estimators=354,
                        max_features="log2", n_jobs=6, random_state=42)
    full = model_params({"estimator": Model()}, regularized=False)
    reduced = model_params({"estimator": Model()}, regularized=True)
    assert full["max_depth"] == 16
    assert reduced["max_depth"] < full["max_depth"]
    assert reduced["min_samples_leaf"] > full["min_samples_leaf"]
    assert reduced["n_jobs"] <= 4
    assert len(ARMS) == 6


def test_spatial_overlap_exact_approx_and_provenance():
    train = pd.DataFrame([
        {"sample_id": "GEO1", "latitude": -43.4879, "longitude": -71.3037,
         "Li_icpms": 28.191, "source_dataset": "GEOROC"},
        {"sample_id": "GEO2", "latitude": -43.5404, "longitude": -72.0516,
         "Li_icpms": 14.079, "source_dataset": "GEOROC"},
    ])
    external = pd.DataFrame([
        {"case_id": "A1015-22", "Latitude": -43.48791666666666,
         "Longitude": -71.30369444444445, "Li_icpms": 28.191},
        {"case_id": "A1015-25", "Latitude": -43.54038888888889,
         "Longitude": -72.05155555555555, "Li_icpms": 14.079},
    ])
    distance = pairwise_distances_km(train, external)
    assert distance.shape == (2, 2)
    assert distance[0, 0] < .01
    assert distance[1, 1] < .01
    audit, independence = audit_external_overlap(train, external)
    assert audit.probable_duplicate.tolist() == [True, True]
    assert independence["probable_duplicate_samples"] == 2
    assert independence["evaluation_independence_approved"] is False


def test_release_gate_fail_closed_incomplete_and_overlapping():
    overlap = {"evaluation_independence_approved": False}
    decision = release_gate(overlap)
    assert decision["approved"] is False
    assert decision["eligible_for_release"] is False
    assert len(decision["reasons"]) >= 2


def test_release_gate_requires_prelocked_confirmatory_metrics():
    external = {"evaluation_independence_approved": True}
    assert release_gate(external, confirmatory_test_locked=False)["approved"] is False
    assert release_gate(external, confirmatory_test_locked=True)["approved"] is False
    failure = release_gate(
        external, confirmatory_test_locked=True,
        independent_roc_auc=.62, independent_average_precision=.32,
        reference_roc_auc=.65, reference_average_precision=.35,
    )
    assert failure["approved"] is False
    success = release_gate(
        external, confirmatory_test_locked=True,
        independent_roc_auc=.68, independent_average_precision=.42,
        reference_roc_auc=.65, reference_average_precision=.35,
    )
    assert success["approved"] is True
    assert success["individual_model_release_allowed"] is False


def test_metrics_requires_both_classes():
    y = np.array([0, 0, 1, 1])
    scores = np.array([.1, .3, .7, .9])
    result = metrics(y, scores)
    assert result["n"] == 4
    assert result["roc_auc"] == 1.
    with pytest.raises(ValueError):
        metrics(np.zeros(4, dtype=int), scores)


def test_invalid_coordinates_rejected():
    train = pd.DataFrame([{"latitude": -99., "longitude": -70.}])
    test = pd.DataFrame([{"Latitude": -43., "Longitude": -71.}])
    with pytest.raises(ValueError, match="limits"):
        pairwise_distances_km(train, test)
