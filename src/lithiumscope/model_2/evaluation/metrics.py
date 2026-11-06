from __future__ import annotations

from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(y_true, probabilities, threshold: float = 0.5) -> dict[str, float]:
    predictions = (probabilities >= threshold).astype(int)
    payload = {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
    }
    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1],
    ).ravel()
    payload["specificity"] = float(tn / (tn + fp)) if tn + fp else float("nan")
    payload["false_positive_rate"] = float(fp / (tn + fp)) if tn + fp else float("nan")
    payload["false_negative_rate"] = float(fn / (fn + tp)) if fn + tp else float("nan")
    payload["log_loss"] = float(log_loss(
        y_true, probabilities, labels=[0, 1],
    ))
    payload["positive_support"] = int(tp + fn)
    payload["negative_support"] = int(tn + fp)
    if len(set(y_true)) > 1:
        payload["roc_auc"] = float(roc_auc_score(y_true, probabilities))
    return payload
