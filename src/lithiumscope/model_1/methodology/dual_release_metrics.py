"""Diagnostics for existing M1 release comparison (no fitting or source edits)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, mean_absolute_error, mean_squared_error, r2_score


def met(y, p):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    if len(y) != len(p) or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Invalid metric vectors")
    return {
        "n": len(y),
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "r2": float(r2_score(y, p)),
        "bias": float(np.mean(p - y)),
        "medae": float(np.median(abs(p - y))),
        "p90ae": float(np.percentile(abs(p - y), 90)),
    }


def four(y, p, cut):
    real = np.asarray(y) >= cut
    guess = np.asarray(p) >= cut
    tn, fp, fn, tp = confusion_matrix(real, guess, labels=[False, True]).ravel()

    def div(a, b):
        return float(a / b) if b else None

    return {
        "VP": int(tp),
        "VN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "n": int(len(real)),
        "n_positive": int(real.sum()),
        "threshold_train_q75": float(cut),
        "precision": div(tp, tp + fp),
        "recall": div(tp, tp + fn),
        "specificity": div(tn, tn + fp),
        "f1": div(2 * tp, 2 * tp + fp + fn),
    }
