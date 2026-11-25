"""Predeclared paired comparisons, with no selection or publication of models."""
from __future__ import annotations

import numpy as np
import pandas as pd


def paired_group_intervals(reference, alternative, *, kind, repeats=500, seed=42):
    """Descriptive paired bootstrap; resample whole groups, not individual rows."""
    keys = ["position", "fold", "group", "truth"]
    if (reference.position.duplicated().any() or alternative.position.duplicated().any()
            or len(reference) != len(alternative) or not len(reference)):
        raise ValueError("Comparison requires exactly the same unique test rows")
    left = reference.sort_values("position").reset_index(drop=True)
    right = alternative.sort_values("position").reset_index(drop=True)
    if not left[keys].equals(right[keys]):
        raise ValueError("Paired test identities or endpoints differ")
    if not isinstance(repeats, int) or repeats < 1:
        raise ValueError("Bootstrap repetitions must be positive")
    y = left.truth.to_numpy(float)
    a, b = left.prediction.to_numpy(float), right.prediction.to_numpy(float)
    if not np.isfinite(np.column_stack((y, a, b))).all():
        raise ValueError("Nonfinite paired observations")
    if left.group.isna().any() or left.group.astype(str).str.strip().eq("").any():
        raise ValueError("Missing bootstrap groups")
    groups = left.group.astype(str).to_numpy()
    clusters = [np.flatnonzero(groups == group) for group in np.unique(groups)]
    if len(clusters) < 2:
        raise ValueError("Bootstrap requires at least two independent group identifiers")
    if kind == "regression":
        def metric(ix):
            e1, e2 = y[ix]-a[ix], y[ix]-b[ix]
            return {"rmse": np.sqrt(np.mean(e2**2))-np.sqrt(np.mean(e1**2)),
                    "mae": np.mean(np.abs(e2))-np.mean(np.abs(e1))}
    elif kind == "classification":
        from sklearn.metrics import roc_auc_score, average_precision_score
        from lithiumscope.core.scientific.classification_metrics import validate_binary_inputs
        validate_binary_inputs(y, a)
        validate_binary_inputs(y, b)
        def metric(ix):
            values = {"brier_score": np.mean((y[ix]-b[ix])**2)-np.mean((y[ix]-a[ix])**2)}
            if np.unique(y[ix]).size == 2:
                values.update(roc_auc=roc_auc_score(y[ix], b[ix])-roc_auc_score(y[ix], a[ix]),
                              average_precision=average_precision_score(y[ix], b[ix])-average_precision_score(y[ix], a[ix]))
            return values
    else:
        raise ValueError("Invalid comparison kind")
    observed = metric(np.arange(len(y)))
    values = {key: [] for key in observed}
    rng = np.random.default_rng(seed)
    for _ in range(repeats):
        ix = np.concatenate([clusters[i] for i in rng.integers(0, len(clusters), len(clusters))])
        for key, value in metric(ix).items():
            values[key].append(float(value))
    result = []
    for key, delta in observed.items():
        valid = values[key]
        # Very few valid resamples cannot support a usable interval.
        enough = len(valid) >= max(2, int(np.ceil(.8*repeats)))
        result.append({"metric": key, "delta_alternative_minus_reference": float(delta),
                       "ci95_low": float(np.quantile(valid, .025)) if enough else None,
                       "ci95_high": float(np.quantile(valid, .975)) if enough else None,
                       "valid_resamples": len(valid), "requested_resamples": repeats,
                       "groups": len(clusters), "rows": len(y),
                       "scope": "exploratory_group_bootstrap_not_external_validation"})
    return result
