"""Descriptive applicability checks based exclusively on TRAIN predictors."""
from __future__ import annotations

import numpy as np


def predictor_support(train, reserved, numeric, categorical):
    outside = np.zeros(len(reserved), dtype=int)
    unknown = np.zeros(len(reserved), dtype=int)
    unsupported = np.zeros(len(reserved), dtype=int)
    for name in numeric:
        x = train[name].to_numpy(float)
        y = reserved[name].to_numpy(float)
        finite = np.isfinite(x)
        observed = np.isfinite(y)
        if finite.any():
            outside += observed & ((y < x[finite].min()) | (y > x[finite].max()))
        else:
            unsupported += observed
    for name in categorical:
        known = set(train[name].dropna())
        unknown += (reserved[name].notna() & ~reserved[name].isin(known)).to_numpy()
    return {"outside_numeric_range":outside, "unseen_categories":unknown,
            "unsupported_numeric_features":unsupported}


def support_summary(train, reserved, numeric, categorical):
    flags = predictor_support(train, reserved, numeric, categorical)
    return {name+"_rows":int((value > 0).sum()) for name,value in flags.items()}
