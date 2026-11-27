from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lithiumscope.core.scientific.classification_metrics import classification_metrics, validate_binary_inputs


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

    # Evaluate all cuts from cumulative confusion counts, not repeated AUC/ECE computations.
    candidates = _candidate_thresholds(scores)
    order = np.argsort(scores, kind="stable")
    sorted_scores, sorted_y = scores[order], y[order]
    below = np.searchsorted(sorted_scores, candidates, side="left")
    prefix = np.r_[0, np.cumsum(sorted_y)]
    fn = prefix[below]
    tp = prefix[-1] - fn
    tn = below - fn
    fp = len(y) - below - tp
    recall = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=tp + fp > 0)
    f1 = np.divide(2*tp, 2*tp + fp + fn, out=np.zeros_like(tp, dtype=float), where=2*tp + fp + fn > 0)
    objectives = {"balanced_accuracy": (recall + specificity)/2,
                  "accuracy": (tp + tn)/len(y), "f1": f1,
                  "precision": precision, "recall": recall, "sensitivity": recall,
                  "specificity": specificity}
    if objective not in objectives:
        raise ValueError(f"Unsupported threshold-dependent objective: {objective}")
    best = max(range(len(candidates)), key=lambda i: (
        float(objectives[objective][i]), float(f1[i]), float(recall[i]),
        -abs(float(candidates[i]) - .5)))
    threshold = float(candidates[best])
    metrics = classification_metrics(y, scores, threshold=threshold)
    return ThresholdSelection(threshold=threshold, objective=objective,
        objective_value=float(metrics[objective]), metrics=metrics,
        candidate_count=int(len(candidates)))
