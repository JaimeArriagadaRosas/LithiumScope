"""Regression contracts for technical PDF without yellow warning callouts."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lithiumscope.prediction.reporting.chart_guides import (
    bar_guides, scatter_guides,
)
from lithiumscope.prediction.reporting.sections.figures import FIGURE_EXPLANATIONS
from lithiumscope.prediction.reporting.styles import build_styles


def _sample():
    return pd.DataFrame({
        "Li_icpms": [1., 2., 3., 4., 5.],
        "Li_icpms_predicted": [1.4, 2.1, 3., 3.8, 4.9],
        "prospectivity_score": [.15, .3, .22, .4, .55],
        "model_1_out_of_training_range_fraction": [0., .1, .2, .25, .3],
        "model_2_out_of_training_range_fraction": [.1, .05, .3, .26, .15],
    })


def test_lithium_scatter_has_identity_and_empirical_fit():
    data = _sample()
    figure, axis = plt.subplots()
    scatter_guides(axis, data, x="Li_icpms", y="Li_icpms_predicted")
    names = [line.get_label() for line in axis.lines]
    assert any("Identidad" in name for name in names)
    assert any("Tendencia lineal" in name for name in names)
    identity = next(line for line in axis.lines if "Identidad" in line.get_label())
    np.testing.assert_allclose(identity.get_xdata(), identity.get_ydata())
    plt.close(figure)


def test_m2_scatter_has_trend_not_false_identity():
    figure, axis = plt.subplots()
    scatter_guides(axis, _sample(), x="Li_icpms", y="prospectivity_score")
    names = [line.get_label() for line in axis.lines]
    assert any("Tendencia lineal" in name for name in names)
    assert not any("Identidad" in name for name in names)
    plt.close(figure)


def test_ood_scatter_has_diagonal_and_two_domain_cuts():
    frame = _sample()
    figure, axis = plt.subplots()
    scatter_guides(
        axis, frame,
        x="model_1_out_of_training_range_fraction",
        y="model_2_out_of_training_range_fraction",
    )
    assert len(axis.lines) == 3
    assert any("Corte OOD" in line.get_label() for line in axis.lines)
    assert not any("Tendencia" in line.get_label() for line in axis.lines)
    plt.close(figure)


def test_bar_references_are_mean_and_domain_cut():
    figure, axis = plt.subplots()
    bar_guides(axis, pd.Series([.1, .3, .2]), is_ood=True)
    assert len(axis.lines) == 2
    assert "Media" in axis.lines[0].get_label()
    assert "Corte OOD" in axis.lines[1].get_label()
    plt.close(figure)


def test_line_fit_is_suppressed_for_constant_or_tiny_samples():
    frame = pd.DataFrame({"Li_icpms": [2., 2.], "prospectivity_score": [.2, .2]})
    figure, axis = plt.subplots()
    scatter_guides(axis, frame, x="Li_icpms", y="prospectivity_score")
    assert len(axis.lines) == 0
    plt.close(figure)


def test_all_report_figures_have_a_short_matching_explanation():
    expected = {
        "li_real_vs_model_1", "li_real_vs_model_2_score",
        "model_1_vs_model_2_score", "model_1_vs_model_2_ood",
        "model_1_absolute_error_by_case", "model_2_score_by_case",
        "model_1_ood_by_case", "model_2_ood_by_case",
    }
    assert set(FIGURE_EXPLANATIONS) == expected


def test_warning_box_style_is_not_available():
    assert "LS_Warning" not in build_styles()
