"""Regression and auxiliary confusion diagnostics: no test-side threshold search."""

import numpy as np
import pytest

from scripts.scientific.m1_dual_release_diagnostics import (
    VERSIONS,
    four,
    met,
)


def test_all_four_confusion_cells_are_grounded():
    d = four([40, 30, 10, 20], [39, 10, 38, 21], 25)
    assert (d["VP"], d["VN"], d["FP"], d["FN"]) == (1, 1, 1, 1)
    assert d["n"] == d["VP"] + d["VN"] + d["FP"] + d["FN"]
    assert d["precision"] == pytest.approx(0.5)
    assert d["recall"] == pytest.approx(0.5)
    assert d["specificity"] == pytest.approx(0.5)


def test_threshold_is_supplied_not_derived_from_outer_test():
    y = [30, 20]
    scores = [24, 26]
    high_cutoff = four(y, scores, 25)
    lower_cutoff = four(y, scores, 22)
    assert high_cutoff["FN"] == 1
    assert lower_cutoff["FN"] == 0
    assert lower_cutoff["VP"] == 1
    assert lower_cutoff["threshold_train_q75"] == 22


def test_missing_positive_class_reports_undefined_recall():
    result = four([1, 2], [1, 1], 100)
    assert result["VN"] == 2
    assert result["VP"] == result["FN"] == result["FP"] == 0
    assert result["recall"] is None


def test_continuous_regression_metrics_are_separate():
    d = met([1, 2, 3, 4], [1, 3, 3, 4])
    assert d["mae"] == pytest.approx(0.25)
    assert d["rmse"] == pytest.approx(0.5)
    assert d["bias"] == pytest.approx(0.25)
    assert d["n"] == 4
    with pytest.raises(ValueError):
        met([1, 2], [np.nan, 2])


def test_evidence_sources_are_frozen_and_distinct():
    old_run, old_sha = VERSIONS["release_1120"]
    new_run, new_sha = VERSIONS["candidate_3420"]
    assert old_run != new_run
    assert old_sha != new_sha
    assert len(old_sha) == len(new_sha) == 64
