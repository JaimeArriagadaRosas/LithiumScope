from __future__ import annotations

import math



def _fmt(value, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/D"
    if not math.isfinite(number):
        return "N/D"
    return f"{number:.{digits}f}"


def _fmt_nonnegative(value, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/D"
    if not math.isfinite(number):
        return "N/D"
    return f"{max(0.0, number):.{digits}f}"


def _corr_strength(value) -> str:
    try:
        magnitude = abs(float(value))
    except (TypeError, ValueError):
        return "no evaluable"
    if not math.isfinite(magnitude):
        return "no evaluable"
    if magnitude < 0.20:
        return "muy débil"
    if magnitude < 0.40:
        return "débil"
    if magnitude < 0.60:
        return "moderada"
    if magnitude < 0.80:
        return "fuerte"
    return "muy fuerte"
