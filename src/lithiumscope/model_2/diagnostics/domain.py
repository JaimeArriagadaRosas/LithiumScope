"""Feature domain profile derived exclusively from a development train split."""
from __future__ import annotations
import hashlib
import numpy as np
import pandas as pd


def training_domain_profile(
    train_features: pd.DataFrame, *, lower: float = 0.01, upper: float = 0.99
) -> dict:
    """Fit descriptive limits on TRAIN rows only, never on a test/demo set."""
    if not (0 <= lower < upper <= 1):
        raise ValueError("Invalid domain quantiles")
    if train_features.empty or train_features.columns.duplicated().any():
        raise ValueError("Domain training frame is empty or has duplicate columns")
    if not all(pd.api.types.is_numeric_dtype(t) for t in train_features.dtypes):
        raise ValueError("Domain features must be numeric")
    arr = train_features.to_numpy(dtype=float)
    if not np.isfinite(arr).all():
        raise ValueError("Domain training features must be finite")
    columns = [str(c) for c in train_features.columns]
    if len(set(columns)) != len(columns):
        raise ValueError("Feature names are not unique")
    return {
        "schema_version": 1,
        "source": "train_only",
        "count": len(train_features),
        "features": columns,
        "lower_quantile": float(lower), "upper_quantile": float(upper),
        "lower": [float(v) for v in np.quantile(arr, lower, axis=0)],
        "upper": [float(v) for v in np.quantile(arr, upper, axis=0)],
        "training_data_sha256": hashlib.sha256(
            np.ascontiguousarray(arr, dtype="<f8").tobytes()
        ).hexdigest(),
    }


def score_domain(features: pd.DataFrame, profile: dict, *, alert_fraction: float = .25) -> pd.DataFrame:
    """Score only; no updating profile with inspected target/test values."""
    if not 0 <= alert_fraction <= 1:
        raise ValueError("Alert fraction must belong to [0,1]")
    columns = list(profile["features"])
    if list(features.columns) != columns:
        raise ValueError("Feature names/order must match the frozen train-only profile")
    x = features.to_numpy(dtype=float)
    low = np.asarray(profile["lower"], dtype=float)
    high = np.asarray(profile["upper"], dtype=float)
    if x.ndim != 2 or x.shape[1] != len(columns):
        raise ValueError("Incorrect feature array")
    invalid = ~np.isfinite(x)
    outlier = invalid | (x < low) | (x > high)
    fractions = outlier.mean(axis=1)
    return pd.DataFrame({
        "ood_fraction": fractions,
        "ood_alert": fractions > alert_fraction,
        "features_outside": outlier.sum(axis=1),
        "has_nonfinite_input": invalid.any(axis=1),
    }, index=features.index)
