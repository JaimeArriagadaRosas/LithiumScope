"""External metrics must reflect only aligned, valid observations."""
import math

import numpy as np
import pandas as pd
import pytest

from lithiumscope.prediction.analysis.metrics import (
    finite_or_none, model_1_external_metrics, model_2_external_metrics,
)


def test_regression_metrics_exclude_nonfinite_pairs():
    frame = pd.DataFrame({
        "Li_icpms": [10., 20., 30., np.inf, 50., "missing"],
        "Li_icpms_predicted": [12., 18., 30., 40., np.inf, 60.],
    })
    before = frame.copy(deep=True)
    metrics = model_1_external_metrics(frame)
    assert metrics["mae"] == pytest.approx(4 / 3)
    assert metrics["rmse"] == pytest.approx(math.sqrt(8 / 3))
    assert metrics["bias"] == pytest.approx(0.)
    assert metrics["r2"] == pytest.approx(.96)
    pd.testing.assert_frame_equal(frame, before)


@pytest.mark.parametrize("frame", [
    pd.DataFrame(),
    pd.DataFrame({"Li_icpms": [10.], "Li_icpms_predicted": [12.]}),
    pd.DataFrame({"Li_icpms": [10., 20.], "Li_icpms_predicted": [np.nan, 12.]}),
])
def test_regression_insufficient_reference_has_no_metrics(frame):
    assert model_1_external_metrics(frame) == {}


def test_classifier_uses_lithium_and_operating_thresholds_separately():
    frame = pd.DataFrame({
        "Li_icpms": [10., 20., 30., 40., np.inf, 50.],
        "prospectivity_score": [.1, .4, .6, .9, .5, np.nan],
    })
    metrics = model_2_external_metrics(frame, threshold_ppm=25., probability_threshold=.7)
    assert metrics["n"] == 4
    assert (metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"]) == (2, 0, 1, 1)
    assert metrics["balanced_accuracy"] == pytest.approx(.75)
    assert metrics["roc_auc"] == pytest.approx(1.)
    assert metrics["threshold"] == .7


def test_classifier_single_class_does_not_claim_discrimination():
    metrics = model_2_external_metrics(
        pd.DataFrame({"Li_icpms": [25., 40.], "prospectivity_score": [.8, .2]}),
        threshold_ppm=20.,
    )
    assert metrics["n"] == 2
    assert math.isnan(metrics["roc_auc"])
    assert math.isnan(metrics["balanced_accuracy"])
    assert metrics["recall"] == .5


@pytest.mark.parametrize("frame", [
    pd.DataFrame({"Li_icpms": [10., 20.]}),
    pd.DataFrame({"Li_icpms": [10., 20.], "prospectivity_score": [.2, np.inf]}),
])
def test_classifier_insufficient_reference_has_no_metrics(frame):
    assert model_2_external_metrics(frame, 20.) == {}


def test_classifier_rejects_out_of_range_scores():
    frame = pd.DataFrame({"Li_icpms": [10., 30.], "prospectivity_score": [.2, 1.1]})
    with pytest.raises(ValueError, match="within"):
        model_2_external_metrics(frame, 20.)


@pytest.mark.parametrize("value,expected", [
    ("1.25", 1.25), (0, 0.), (None, None), ("invalid", None),
    (float("nan"), None), (float("inf"), None),
])
def test_finite_values(value, expected):
    assert finite_or_none(value) == expected
