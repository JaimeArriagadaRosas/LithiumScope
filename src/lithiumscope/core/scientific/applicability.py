"""Shared marginal applicability contract; no evidence is not in-domain evidence."""
from __future__ import annotations

import numpy as np


def complete_range_profile(profile: dict, expected) -> bool:
    if not isinstance(profile, dict) or not expected or set(profile) != set(expected):
        return False
    try:
        return all(np.isfinite([float(v["q01"]), float(v["q99"])]).all()
                   and float(v["q01"]) <= float(v["q99"]) for v in profile.values())
    except (KeyError, TypeError, ValueError):
        return False
