"""Regression contracts for the controlled M2 scientific remediation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from lithiumscope.model_2.evaluation.generalization import _metrics
from lithiumscope.model_2.evaluation.independence import audit_external_overlap
from lithiumscope.model_2.evaluation.release_gate import (
    ExternalGatePolicy,
    assess_release,
)
from lithiumscope.model_2.methodology.regularization_study import (
    _apply_calibrator,
    _calibrator,
    _inner_scores,
    _metrics as experiment_metrics,
)


def test_geographic_and_lithium_duplicate_check():
    external = pd.DataFrame({
        "case_id": ["A1015-22", "unique"],
        "Latitude": [-43.4879167, -34.0],
        "Longitude": [-71.3036944, -70.0],
        "Li_icpms": [28.191, 2.5],
    })
    training = pd.DataFrame({
        "sample_id": ["GEOROC_one", "GEOROC_two"],
        "latitude": [-43.4879, -43.48792],
        "longitude": [-71.3037, -71.303695],
        "Li_icpms": [28.191, 0.5],
        "source_dataset": ["GEOROC", "GEOROC"],
    })
    result = audit_external_overlap(external, training)
    assert result["possible_duplicate_cases"] == ["A1015-22"]
    assert result["possible_duplicate_count"] == 1
    assert result["rows"][0]["nearest_distance_m"] < 10
    assert not result["external_independence_verified"]


def test_nearby_nonmatching_li_is_not_a_confirmed_duplicate():
    external = pd.DataFrame({
        "case_id": ["different_lab"],
        "Latitude": [-43.5],
        "Longitude": [-71.3],
        "Li_icpms": [70.0],
    })
    training = pd.DataFrame({
        "sample_id": ["one"],
        "latitude": [-43.5],
        "longitude": [-71.3],
        "Li_icpms": [4.5],
    })
    report = audit_external_overlap(external, training)
    assert report["same_site_cases"] == ["different_lab"]
    assert report["possible_duplicate_count"] == 0
    assert report["external_independence_verified"] is False


def test_bad_geography_fails_closed():
    with pytest.raises(ValueError, match="Missing geographic"):
        audit_external_overlap(
            pd.DataFrame({"Li_icpms": [10]}),
            pd.DataFrame({"latitude": [1], "longitude": [2], "Li_icpms": [4]}),
        )


def test_release_never_approves_reused_ten_case_demo():
    evidence = {
        "n": 10, "positives": 2, "negatives": 8, "regions": 1,
        "external_independence_verified": False,
        "holdout_unconsulted_during_model_selection": False,
        "sample_identity_and_provenance_verified": False,
        "possible_duplicate_count": 2,
        "labels_field_verified": False,
        "candidate_auc": 0.5625, "reference_auc": 0.6875,
        "candidate_ap": 0.6111, "reference_ap": 0.6428,
        "candidate_brier": 0.15, "reference_brier": 0.161,
        "paired_auc_gain_ci_lower": -0.4,
    }
    decision = assess_release(evidence)
    assert not decision["scientific_release_evidence_pass"]
    assert "possible_training_or_holdout_duplicates" in decision["violations"]
    assert "insufficient_external_samples" in decision["violations"]
    assert not decision["automatic_model_activation"]


def test_release_requires_review_even_on_synthetic_pass():
    synthetic = {
        "n": 120, "positives": 25, "negatives": 95, "regions": 3,
        "external_independence_verified": True,
        "holdout_unconsulted_during_model_selection": True,
        "sample_identity_and_provenance_verified": True,
        "possible_duplicate_count": 0,
        "labels_field_verified": True,
        "candidate_auc": .70, "reference_auc": .66,
        "candidate_ap": .58, "reference_ap": .54,
        "candidate_brier": .18, "reference_brier": .175,
        "paired_auc_gain_ci_lower": -.01,
    }
    decision = assess_release(synthetic, policy=ExternalGatePolicy())
    assert decision["scientific_release_evidence_pass"]
    assert not decision["automatic_model_activation"]
    assert decision["requires_explicit_human_approval"]


def test_inner_spatial_cv_and_platt_score_are_complete():
    # Five groups, each with both classes; all inner training folds have two classes.
    generator = np.random.default_rng(42)
    x = pd.DataFrame(generator.normal(size=(50, 4)))
    groups = pd.Series(np.repeat(np.arange(5), 10))
    y = np.tile([0, 1], 25)
    forest = RandomForestClassifier(
        n_estimators=12, max_depth=3, min_samples_leaf=2,
        random_state=42, n_jobs=1,
    )
    score = _inner_scores(forest, x, y, groups, 42)
    assert np.isfinite(score).all()
    assert ((score >= 0) & (score <= 1)).all()
    calibrated = _apply_calibrator(_calibrator(score, y), score)
    assert len(calibrated) == len(y)
    assert np.isfinite(calibrated).all()
    assert 0 <= calibrated.min() <= calibrated.max() <= 1
    assert experiment_metrics(y, calibrated, 0.5)["n"] == 50
    assert _metrics(y, score)["n"] == 50
