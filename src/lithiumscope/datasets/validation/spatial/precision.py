"""Reported geographic extent and uncertainty, not inferred sampling accuracy."""
from __future__ import annotations

import numpy as np
import pandas as pd


def assess_precision(frame: pd.DataFrame, *, max_uncertainty_m: float | None = None) -> pd.DataFrame:
    if max_uncertainty_m is not None and (not np.isfinite(max_uncertainty_m) or max_uncertainty_m < 0):
        raise ValueError("Invalid maximum uncertainty")
    result = pd.DataFrame(index=frame.index)
    status = pd.Series("unknown", index=frame.index, dtype=str)
    uncertainty = pd.Series(np.nan, index=frame.index, dtype=float)
    invalid = pd.Series(False, index=frame.index)
    for name in ("coordinate_uncertainty_m", "precision_m"):
        if name not in frame:
            continue
        raw = frame[name]
        numeric = pd.to_numeric(raw, errors="coerce")
        valid = np.isfinite(numeric) & numeric.ge(0)
        invalid |= raw.notna() & ~valid
        # Use the largest reported value, not the most optimistic alias.
        uncertainty = pd.concat([uncertainty, numeric.where(valid)], axis=1).max(axis=1)
    bounds_issue = pd.Series(False, index=frame.index)
    for axis, limit in (("Latitude", 90), ("Longitude", 180)):
        lower, upper = f"{axis}_min", f"{axis}_max"
        if lower not in frame and upper not in frame:
            continue
        a = pd.to_numeric(frame[lower], errors="coerce") if lower in frame else pd.Series(np.nan, index=frame.index)
        b = pd.to_numeric(frame[upper], errors="coerce") if upper in frame else pd.Series(np.nan, index=frame.index)
        reported = (frame[lower].notna() if lower in frame else False) | (frame[upper].notna() if upper in frame else False)
        valid = a.between(-limit, limit) & b.between(-limit, limit)
        # A reversed longitude could cross the antimeridian; do not guess.
        ordered = valid & a.le(b)
        span = (b - a).where(ordered)
        result[f"{axis.lower()}_reported_span_deg"] = span
        bounds_issue |= reported & ~ordered
        if axis in frame:
            center = pd.to_numeric(frame[axis], errors="coerce")
            bounds_issue |= ordered & center.notna() & ~(center.ge(a) & center.le(b))
    status.loc[uncertainty.notna()] = "reported"
    if max_uncertainty_m is not None:
        status.loc[uncertainty.gt(max_uncertainty_m)] = "exceeds_policy"
    status.loc[invalid] = "invalid_uncertainty"
    status.loc[bounds_issue] = "bounds_review"
    result["precision_status"] = status
    result["reported_uncertainty_m"] = uncertainty
    result["bounds_review"] = bounds_issue
    return result
