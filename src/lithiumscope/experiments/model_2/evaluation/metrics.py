"""Historical grouped study scoring; preserve its metrics and threshold."""
import numpy as np
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, brier_score_loss,
    f1_score, precision_score, recall_score, roc_auc_score,
)

def expected_calibration_error(
    labels, probabilities, *, bins: int = 10,
) -> float:
    labels = np.asarray(labels, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    if labels.shape != probabilities.shape or not len(labels):
        raise ValueError("Invalid labels / probabilities")
    if not np.isfinite(probabilities).all():
        raise ValueError("Probabilities contain non-finite values")
    edges = np.linspace(0, 1, bins+1)
    result = 0.0
    for idx in range(bins):
        mask = ((probabilities >= edges[idx])
                & (probabilities < edges[idx+1]
                   if idx < bins - 1 else probabilities <= edges[idx+1]))
        if mask.any():
            result += float(mask.mean()) * abs(
                float(labels[mask].mean()) - float(probabilities[mask].mean())
            )
    return float(result)


def score_metrics(y, p) -> dict[str, float]:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    if not np.isin(y, [0, 1]).all() or len(np.unique(y)) < 2:
        raise ValueError("Heldout fold must contain both classes")
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "balanced_accuracy": float(balanced_accuracy_score(y, p>=0.5)),
        "f1": float(f1_score(y, p>=0.5, zero_division=0)),
        "precision": float(precision_score(y, p>=0.5, zero_division=0)),
        "recall": float(recall_score(y, p>=0.5, zero_division=0)),
        "brier": float(brier_score_loss(y,p)),
        "ece_10": expected_calibration_error(y,p),
    }
