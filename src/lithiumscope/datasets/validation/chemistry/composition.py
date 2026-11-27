"""Physical composition checks; no implicit outlier deletion or imputation."""
from __future__ import annotations

import numpy as np
import pandas as pd


def assess_composition(frame: pd.DataFrame, columns: tuple[str, ...], *,
                       total_bounds: tuple[float, float] | None = None) -> pd.DataFrame:
    if not columns or len(set(columns)) != len(columns):
        raise ValueError("Composition requires distinct oxide columns")
    if not frame.columns.is_unique:
        raise ValueError("Ambiguous composition columns")
    if total_bounds is not None and (len(total_bounds) != 2 or
            not all(np.isfinite(total_bounds)) or not 0 <= total_bounds[0] <= total_bounds[1]):
        raise ValueError("Invalid composition total bounds")
    raw = frame.reindex(columns=list(columns))
    numeric = raw.apply(pd.to_numeric, errors="coerce")
    finite = np.isfinite(numeric)
    physical = finite & numeric.ge(0) & numeric.le(100)
    present_invalid = raw.notna() & ~physical
    complete = physical.all(axis=1)
    total = numeric.where(physical).sum(axis=1, min_count=len(columns))
    status = pd.Series("incomplete", index=frame.index, dtype=str)
    status.loc[complete] = "complete"
    if total_bounds is not None:
        status.loc[complete & ~total.between(*total_bounds)] = "total_outside_policy"
    status.loc[present_invalid.any(axis=1)] = "invalid_component"
    return pd.DataFrame({"composition_status": status, "oxide_total_wt_pct": total,
                         "valid_oxide_count": physical.sum(axis=1),
                         "invalid_oxide_count": present_invalid.sum(axis=1)}, index=frame.index)
