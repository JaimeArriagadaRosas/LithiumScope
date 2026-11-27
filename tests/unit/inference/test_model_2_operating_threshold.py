import numpy as np

from lithiumscope.model_2.evaluation.thresholds import (
    select_operating_threshold,
)


def test_operating_threshold_is_learned_from_oof_scores():
    y_true = np.asarray([0, 0, 0, 1, 1, 1])
    scores = np.asarray([0.05, 0.10, 0.20, 0.30, 0.40, 0.45])

    selection = select_operating_threshold(
        y_true,
        scores,
        objective="balanced_accuracy",
    )

    assert selection.threshold < 0.5
    assert selection.metrics["balanced_accuracy"] == 1.0
    assert selection.metrics["recall"] == 1.0


def test_operating_threshold_falls_back_without_two_classes():
    y_true = np.asarray([0, 0, 0])
    scores = np.asarray([0.1, 0.2, 0.3])

    selection = select_operating_threshold(
        y_true,
        scores,
        fallback=0.5,
    )

    assert selection.threshold == 0.5
