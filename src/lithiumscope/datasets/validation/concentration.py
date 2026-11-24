"""Shared physical concentration contract, independent of model-specific filtering."""
from __future__ import annotations

import numpy as np
import pandas as pd


def valid_concentration(values: pd.Series, *, zero_policy: str = "retain") -> tuple[pd.Series, pd.Series, dict]:
    """Preserve finite nonnegative values; retain/exclude zero explicitly.

    A retained zero is not evidence of a detected concentration; its source
    semantics must be audited before a certified scientific comparison.
    """
    if zero_policy not in {"retain", "exclude"}:
        raise ValueError(f"Unsupported concentration zero policy: {zero_policy}")
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    finite = pd.Series(np.isfinite(numeric), index=values.index)
    missing = numeric.isna()
    negative = finite & (numeric < 0)
    zero = finite & (numeric == 0)
    mask = finite & ~negative
    if zero_policy == "exclude":
        mask &= ~zero
    audit = {
        "input_rows": len(values),
        "valid_rows": int(mask.sum()),
        "excluded_negative": int(negative.sum()),
        "excluded_nonfinite": int((~finite & ~missing).sum()),
        "excluded_missing_or_unparseable": int(missing.sum()),
        "zero_count": int(zero.sum()),
        "excluded_zero": int(zero.sum()) if zero_policy == "exclude" else 0,
        "zero_policy": zero_policy,
    }
    return numeric, mask, audit
