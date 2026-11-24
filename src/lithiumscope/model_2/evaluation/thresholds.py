from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lithiumscope.model_2.evaluation.metrics import classification_metrics, validate_binary_inputs


@dataclass(frozen=True)
class ThresholdSelection:
    threshold: float
    objective: str
    objective_value: float
    metrics: dict[str, float]
    candidate_count: int
    state: str = "selected"


def _candidate_thresholds(probabilities: np.ndarray) -> np.ndarray:
    finite = np.unique(
        np.clip(
            probabilities[np.isfinite(probabilities)],
            0.0,
            1.0,
        )
    )
    if finite.size == 0:
        return np.asarray([0.5], dtype=float)
    if finite.size == 1:
        return np.unique(
            np.asarray(
                [0.0, float(finite[0]), 0.5, 1.0],
                dtype=float,
            )
        )
    midpoints = (finite[:-1] + finite[1:]) / 2.0
    return np.unique(
        np.concatenate(
            [
                np.asarray([0.0, 0.5, 1.0], dtype=float),
                finite,
                midpoints,
            ]
        )
    )


def select_operating_threshold(
    y_true,
    probabilities,
    *,
    objective: str = "balanced_accuracy",
    fallback: float = 0.5,
) -> ThresholdSelection:
    y,scores=validate_binary_inputs(y_true,probabilities,require_both=False)
    if not np.isfinite(fallback) or not 0<=fallback<=1:
        raise ValueError("Fallback threshold outside [0,1]")
    if np.unique(y).size < 2:
        return ThresholdSelection(threshold=float(fallback),objective=objective,
            objective_value=float("nan"),metrics={},candidate_count=0,state="unavailable_single_class")

    candidates = _candidate_thresholds(scores)
    best = None
    for threshold in candidates:
        metrics = classification_metrics(
            y,
            scores,
            threshold=float(threshold),
        )
        if objective not in metrics:
            raise ValueError(
                f"Unsupported threshold objective: {objective}"
            )
        key = (
            float(metrics[objective]),
            float(metrics.get("f1", 0.0)),
            float(metrics.get("recall", 0.0)),
            -abs(float(threshold) - 0.5),
        )
        if best is None or key > best[0]:
            best = (
                key,
                float(threshold),
                metrics,
            )

    assert best is not None
    return ThresholdSelection(
        threshold=best[1],
        objective=objective,
        objective_value=float(best[2][objective]),
        metrics={
            key: float(value)
            for key, value in best[2].items()
        },
        candidate_count=int(len(candidates)),
    )
