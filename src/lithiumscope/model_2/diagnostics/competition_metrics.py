"""Confusion matrices and per-class spatial diagnostic for M2 OOF data.

A common fixed decision threshold 0.5 is descriptive, NOT a tuned operating
threshold. No outer-test label is used to choose a threshold or model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def binary_diagnostic(y_true, probabilities, *, threshold: float = 0.5) -> dict:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    if (len(y) == 0 or y.shape != p.shape or
        not np.isin(y, [0, 1]).all() or
        not np.isfinite(p).all() or
        np.any((p < 0) | (p > 1)) or
        not np.isfinite(threshold) or not 0 < threshold < 1):
        raise ValueError("Invalid diagnostic labels, scores or fixed threshold")
    guess = p >= threshold
    positive = y == 1
    tp = int(np.sum(guess & positive))
    fp = int(np.sum(guess & ~positive))
    tn = int(np.sum(~guess & ~positive))
    fn = int(np.sum(~guess & positive))

    def rate(num, denom):
        return float(num / denom) if denom else None

    positive_values = p[positive]
    negative_values = p[~positive]
    return {
        "n": len(y), "actual_positive": tp + fn,
        "actual_negative": tn + fp, "VP": tp, "VN": tn, "FP": fp, "FN": fn,
        "threshold": float(threshold),
        "precision": rate(tp, tp + fp),
        "recall_sensitivity": rate(tp, tp + fn),
        "specificity": rate(tn, tn + fp),
        "false_positive_rate": rate(fp, fp + tn),
        "false_negative_rate": rate(fn, fn + tp),
        "negative_predictive_value": rate(tn, tn + fn),
        "f1": rate(2 * tp, 2 * tp + fp + fn),
        "balanced_accuracy": (
            0.5 * (tp / (tp + fn) + tn / (tn + fp))
            if tp + fn and tn + fp else None
        ),
        "roc_auc": float(roc_auc_score(y, p)) if len(set(y)) == 2 else None,
        "average_precision": (
            float(average_precision_score(y, p))
            if (tp + fn) and (tn + fp) else None
        ),
        "brier": float(brier_score_loss(y, p)),
        "brier_positive": (
            float(np.mean((1 - positive_values) ** 2))
            if len(positive_values) else None
        ),
        "brier_negative": (
            float(np.mean(negative_values ** 2))
            if len(negative_values) else None
        ),
        "mean_score": float(p.mean()),
        "prevalence": float(y.mean()),
    }


def sample_outcomes(y: np.ndarray, score: np.ndarray, *, threshold=0.5):
    y = np.asarray(y, int)
    score = np.asarray(score, float)
    return np.select([
        (y == 1) & (score >= threshold),
        (y == 0) & (score < threshold),
        (y == 0) & (score >= threshold),
        (y == 1) & (score < threshold),
    ], ["VP", "VN", "FP", "FN"], default="INVALID")


def spatial_bootstrap_difference(y, first, second, groups, *, seed=42, n=200):
    """Paired group bootstrap CI for current-minus-reference metrics."""
    y = np.asarray(y, int)
    first = np.asarray(first, float)
    second = np.asarray(second, float)
    groups = np.asarray(groups, str)
    if len(y) != len(groups) or first.shape != y.shape or second.shape != y.shape:
        raise ValueError("Spatial bootstrap must be row aligned")
    distinct = np.unique(groups)
    if len(distinct) < 5:
        raise ValueError("At least 5 independent groups needed for CI")
    rng = np.random.default_rng(seed)
    indices = [np.flatnonzero(groups == g) for g in distinct]
    records = {"roc_auc": [], "average_precision": [], "brier": []}
    metric = {
        "roc_auc": roc_auc_score,
        "average_precision": average_precision_score,
        "brier": brier_score_loss,
    }
    for _ in range(n):
        sampled = np.concatenate(
            [indices[k] for k in rng.integers(len(indices), size=len(indices))]
        )
        yy = y[sampled]
        if len(np.unique(yy)) != 2:
            continue
        for key, fn in metric.items():
            records[key].append(
                float(fn(yy, second[sampled]) - fn(yy, first[sampled]))
            )
    return {
        key: {
            "delta": float(fn(y, second) - fn(y, first)),
            "ci95": [
                float(v) for v in np.quantile(records[key], [.025, .975])
            ] if len(records[key]) >= max(30, int(n * .8)) else None,
            "iterations_valid": len(records[key]),
        }
        for key, fn in metric.items()
    }


def reliability_rows(name: str, y: np.ndarray, scores: np.ndarray):
    """Summaries from fixed absolute score bins, not learned from held-out y."""
    for lower, upper in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        membership = (scores >= lower) & (
            scores <= upper if upper == 1 else scores < upper
        )
        if membership.any():
            yield {
                "algorithm": name, "bin_low": lower, "bin_high": upper,
                "n": int(membership.sum()),
                "mean_score": float(scores[membership].mean()),
                "positive_rate": float(y[membership].mean()),
                "calibration_gap": float(
                    scores[membership].mean() - y[membership].mean()
                ),
            }
