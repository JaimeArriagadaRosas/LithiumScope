"""Paired spatial cluster bootstrap used by the historical M1 comparisons."""
from __future__ import annotations

import numpy as np
from lithiumscope.core.scientific.regression_metrics import regression_metrics


def cluster_bootstrap_improvement(
    y, a, b, groups, *, repeats: int = 400, seed: int = 42,
) -> dict[str, dict[str, float]]:
    y = np.asarray(y, dtype=float)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    g = np.asarray(groups).astype(str)
    if not (len(y) == len(a) == len(b) == len(g)):
        raise ValueError("Lengths must match")
    labels = np.unique(g)
    if len(labels) < 2:
        raise ValueError("At least two spatial groups are required")
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {"mae_gain": [], "rmse_gain": []}
    slices = {x: np.flatnonzero(g == x) for x in labels}
    for _ in range(repeats):
        selected = rng.choice(labels, len(labels), replace=True)
        idx = np.concatenate([slices[x] for x in selected])
        ay = a[idx] - y[idx]
        by = b[idx] - y[idx]
        values["mae_gain"].append(float(np.mean(abs(ay)) - np.mean(abs(by))))
        values["rmse_gain"].append(
            float(np.sqrt(np.mean(ay**2)) - np.sqrt(np.mean(by**2)))
        )
    return {
        key: {
            "low_95": float(np.percentile(x, 2.5)),
            "median": float(np.median(x)),
            "high_95": float(np.percentile(x, 97.5)),
            "positive_fraction": float(np.mean(np.asarray(x) > 0)),
        }
        for key, x in values.items()
    }


def comparison_metrics(y, prediction) -> dict[str, float]:
    metrics = regression_metrics(y, prediction)
    return {name: getattr(metrics, name) for name in ("mae", "rmse", "bias", "p90_ae")}
