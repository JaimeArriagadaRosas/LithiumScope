"""Multivariate development-domain diagnostic, fitted only on reference predictors."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


def _matrix(frame, numeric):
    if not set(numeric) <= set(frame) or len(set(numeric)) != len(numeric):
        raise ValueError("Missing or duplicated domain feature schema")
    values = frame[list(numeric)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float, copy=True)
    values[~np.isfinite(values)] = np.nan
    return values


def _distances(values, center, scale, location, precision):
    filled = np.where(np.isfinite(values), values, center)
    standardized = (filled-center)/scale-location
    return np.sqrt(np.maximum(0., np.einsum("ij,jk,ik->i", standardized, precision, standardized)))


def fit_domain_profile(frame, numeric, categorical, *, min_observed=.75, quantile=.99):
    if not numeric or not 0 < min_observed <= 1 or not 0 < quantile < 1:
        raise ValueError("Invalid domain policy")
    values = _matrix(frame, numeric)
    if np.isnan(values).all(axis=0).any():
        raise ValueError("Domain reference contains empty numeric predictors")
    eligible = np.isfinite(values).mean(axis=1) >= min_observed
    if eligible.sum() < 3:
        raise ValueError("Insufficient development rows for domain profile")
    reference = values[eligible]
    if np.isnan(reference).all(axis=0).any():
        raise ValueError("Eligible reference has empty features")
    center = np.nanmedian(reference, axis=0)
    q25, q75 = np.nanquantile(reference, [.25, .75], axis=0)
    scale = q75-q25
    fallback = np.nanstd(reference, axis=0)
    scale = np.where(scale > 0, scale, np.where(fallback > 0, fallback, 1.))
    filled = np.where(np.isfinite(reference), reference, center)
    covariance = LedoitWolf().fit((filled-center)/scale)
    distances = _distances(reference, center, scale, covariance.location_, covariance.precision_)
    categories = {c: sorted(frame.loc[eligible, c].dropna().astype(str).unique().tolist()) for c in categorical}
    return {"version": 1, "numeric": list(numeric), "categorical": categories,
            "median": center.tolist(), "scale": scale.tolist(), "location": covariance.location_.tolist(),
            "precision": covariance.precision_.tolist(), "distance_limit": float(np.quantile(distances, quantile)),
            "quantile": quantile, "min_observed": min_observed, "reference_rows": int(eligible.sum()),
            "interpretation": "development_reference_not_calibrated_ood_probability"}


def evaluate_domain(frame, profile):
    if profile.get("version") != 1:
        raise ValueError("Unsupported multivariate profile")
    values = _matrix(frame, profile["numeric"])
    observed = np.isfinite(values).mean(axis=1)
    d = len(profile["numeric"])
    center, scale, loc, precision = [np.asarray(profile[k], float) for k in ("median", "scale", "location", "precision")]
    limit, minimum = float(profile["distance_limit"]), float(profile["min_observed"])
    if (any(v.shape != (d,) for v in (center, scale, loc)) or precision.shape != (d,d)
            or not all(np.isfinite(v).all() for v in (center, scale, loc, precision))
            or (scale <= 0).any() or not np.isfinite(limit) or limit < 0 or not 0 < minimum <= 1):
        raise ValueError("Invalid multivariate profile values")
    distance = _distances(values, center, scale, loc, precision)
    unseen = np.zeros(len(frame), dtype=int)
    for col, known in profile["categorical"].items():
        if col not in frame:
            raise ValueError("Missing expected category column")
        unseen += (frame[col].notna() & ~frame[col].astype(str).isin(known)).to_numpy(int)
    status = np.where(observed < minimum, "INSUFFICIENT_INPUT",
                      np.where((distance > limit) | (unseen > 0), "OUT_OF_DOMAIN", "OK"))
    return pd.DataFrame({"multivariate_distance": distance, "numeric_observed_fraction": observed,
                         "unseen_category_count": unseen, "multivariate_warning": status}, index=frame.index)
