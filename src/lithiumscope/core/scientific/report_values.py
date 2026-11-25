"""Scalar scientific report values, independent of any model or renderer."""
from __future__ import annotations
import math

def operating_classification(score, threshold):
    try:
        s,t=float(score),float(threshold)
    except (ValueError,TypeError):
        return "N/D"
    if not (math.isfinite(s) and math.isfinite(t) and 0<=s<=1 and 0<=t<=1):
        return "N/D"
    return "POSITIVA" if s>=t else "NEGATIVA"


def physical_interval(low, high):
    try:
        lo,hi=float(low),float(high)
    except (ValueError,TypeError):
        return None,None
    if not (math.isfinite(lo) and math.isfinite(hi) and hi>=max(0.,lo)):
        return None,None
    return max(0.,lo),hi


def nonnegative_interval(low, high, digits: int = 3) -> str:
    """Historical formatted interval helper; no renderer dependency."""
    lo,hi=physical_interval(low,high)
    if lo is None:
        return "N/D"
    return f"{lo:.{digits}f} a {hi:.{digits}f} ppm"
