"""Six-algorithm competition diagnostics and train-only calibration contracts."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lithiumscope.model_2.training.calibration_policy import (
    calibration_policy, fit_spatially_calibrated,
)
from lithiumscope.model_2.training.diagnostic_metrics import (
    binary_diagnostic, sample_outcomes, spatial_bootstrap_difference,
)


def test_confusion_matrix_all_four_cells_and_rates():
    labels = np.array([1, 1, 0, 0, 1, 0])
    values = np.array([.8, .2, .9, .1, .6, .4])
    result = binary_diagnostic(labels, values)
    assert (result["VP"], result["VN"], result["FP"], result["FN"]) == (2, 2, 1, 1)
    assert result["n"] == sum(result[x] for x in ("VP", "VN", "FP", "FN"))
    assert result["precision"] == pytest.approx(2 / 3)
    assert result["recall_sensitivity"] == pytest.approx(2 / 3)
    assert result["specificity"] == pytest.approx(2 / 3)
    assert result["false_negative_rate"] == pytest.approx(1 / 3)
    assert result["brier_positive"] is not None
    assert result["brier_negative"] is not None
    assert sample_outcomes(labels, values).tolist() == [
        "VP", "FN", "FP", "VN", "VP", "VN",
    ]


def test_all_negative_outcomes_have_undefined_recall_not_fabricated():
    result = binary_diagnostic([0, 0, 0], [.2, .1, .3])
    assert result["VN"] == 3 and result["VP"] == 0
    assert result["recall_sensitivity"] is None
    assert result["false_negative_rate"] is None
    assert result["brier_positive"] is None


def test_rejects_nonfinite_scores_or_invalid_confusion_threshold():
    with pytest.raises(ValueError):
        binary_diagnostic([1, 0], [.5, np.nan])
    with pytest.raises(ValueError):
        binary_diagnostic([1, 0], [.5, .6], threshold=0)
    with pytest.raises(ValueError):
        binary_diagnostic([1, 0], [.5, 1.1])


def test_spatial_bootstrap_rejects_unaligned_predictions():
    with pytest.raises(ValueError, match="row aligned"):
        spatial_bootstrap_difference(
            [1, 0], [.1, .2, .3], [.5, .4], ["a", "b"]
        )


def test_spatial_bootstrap_pairs_each_group_and_returns_intervals():
    groups = np.repeat(np.arange(10), 10).astype(str)
    labels = np.tile([0, 1] * 5, 10)
    baseline = np.tile([.35, .65] * 5, 10)
    candidate = np.tile([.2, .8] * 5, 10)
    output = spatial_bootstrap_difference(
        labels, baseline, candidate, groups, n=40
    )
    assert output["roc_auc"]["ci95"] is not None
    assert output["brier"]["delta"] < 0
    assert output["brier"]["ci95"][1] < 0


def test_calibration_policy_is_versioned_and_rejects_unrecognized():
    assert calibration_policy({}) == "legacy"
    assert calibration_policy({"calibration_policy": "spatial_sigmoid"}) == "spatial_sigmoid"
    with pytest.raises(ValueError, match="Unknown"):
        calibration_policy({"calibration_policy": "autoselect_based_on_test"})


def _spectral_synthetic():
    rng = np.random.default_rng(18)
    groups = pd.Series(np.repeat(np.arange(18).astype(str), 4))
    n = len(groups)
    x = pd.DataFrame(rng.normal(size=(n, 6)), columns=[
        "red_mean", "green_mean", "blue_mean",
        "nir_mean", "swir16_mean", "swir22_mean",
    ])
    # Every group has both classes: no scientific split gets a one-class group.
    y = pd.Series(np.tile([0, 1, 0, 1], 18))
    return x, y, groups


def test_calibration_never_mixes_spatial_groups_and_fits_rf(monkeypatch):
    from lithiumscope.model_2.training import calibration_policy as mod
    x, y, groups = _spectral_synthetic()
    observed = {}
    original = mod.create_model

    def factory(*args, **kwargs):
        observed["uncalibrated"] = kwargs.get("calibrated") is False
        return original(*args, **kwargs)

    monkeypatch.setattr(mod, "create_model", factory)
    fitted = fit_spatially_calibrated(
        "random_forest", None, 42,
        {"n_estimators": 20, "n_jobs": 1, "max_depth": 3},
        x, y, groups, inner_folds=3,
    )
    assert observed["uncalibrated"] is True
    scores = fitted.predict_proba(x.iloc[:3])[:, 1]
    assert scores.shape == (3,) and np.isfinite(scores).all()


def test_spatial_calibration_rejects_missing_groups_and_missing_class():
    x, y, groups = _spectral_synthetic()
    with pytest.raises(ValueError, match="spatial groups"):
        fit_spatially_calibrated(
            "random_forest", None, 42, {}, x, y, None, inner_folds=3
        )
    with pytest.raises(ValueError, match="two training classes"):
        fit_spatially_calibrated(
            "random_forest", None, 42, {}, x, y * 0, groups, inner_folds=3
        )


def test_calibration_policy_handles_two_distinct_algorithms():
    x, y, groups = _spectral_synthetic()
    for algorithm, params in [
        ("extra_trees", {"n_estimators": 10, "n_jobs": 1}),
        ("svm_rbf", {"C": 0.5}),
    ]:
        fitted = fit_spatially_calibrated(
            algorithm, None, 42, params, x, y, groups, inner_folds=3
        )
        assert np.isfinite(fitted.predict_proba(x.iloc[:4])).all()


def test_historical_release_reference_all_algorithms_no_automatic_promotion(tmp_path):
    from lithiumscope.model_2.training.diagnostic_release_comparison import (
        compare_to_release_protocol, REFERENCE,
    )
    tables = tmp_path / "tables"
    tables.mkdir()
    n = 706
    common = {
        "test_index": np.arange(n),
        "fold": np.arange(n) % 5 + 1,
        "sample_id": [f"S{i:04d}" for i in range(n)],
        "spatial_group": [f"G{i % 32}" for i in range(n)],
        "y_true": np.arange(n) % 4 == 0,
    }
    base = pd.DataFrame({
        **common,
        "arm": REFERENCE,
        "score": np.where(common["y_true"], .6, .3),
    })
    candidate = pd.DataFrame({
        **common,
        "arm": "catboost",
        "score": np.where(common["y_true"], .75, .2),
    })
    pd.concat([base, candidate]).to_csv(
        tables / "old_release_protocol_paired_706.csv", index=False
    )
    info = compare_to_release_protocol(tables, tmp_path / "out", repeats=40)
    assert info["historical_test_cases"] == 706
    assert info["algorithm_count"] == 1
    assert info["release_pass"] is False
    report = pd.read_csv(
        tmp_path / "out" / "historical_release_comparison_six_algorithms.csv"
    )
    assert report.iloc[0].auc_gain == pytest.approx(0)
    assert report.iloc[0].brier_change < 0
    assert not bool(report.iloc[0].external_release_approved)


def test_release_reference_misaligned_samples_is_rejected(tmp_path):
    from lithiumscope.model_2.training.diagnostic_release_comparison import (
        compare_to_release_protocol, REFERENCE,
    )
    t = tmp_path / "tables"
    t.mkdir()
    n = 706
    common = {
        "test_index": np.arange(n),
        "fold": np.arange(n) % 5 + 1,
        "sample_id": [f"S{i}" for i in range(n)],
        "spatial_group": [f"G{i % 12}" for i in range(n)],
        "y_true": np.arange(n) % 2,
    }
    base = pd.DataFrame({**common, "arm": REFERENCE, "score": .3})
    newer = base.copy()
    newer["arm"] = "extra_trees"
    newer.loc[3, "sample_id"] = "incorrect"
    pd.concat([base, newer]).to_csv(
        t / "old_release_protocol_paired_706.csv", index=False
    )
    with pytest.raises(ValueError, match="sample_id"):
        compare_to_release_protocol(t, tmp_path / "out", repeats=40)
