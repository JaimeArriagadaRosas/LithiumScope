from __future__ import annotations

import math
from xml.sax.saxutils import escape


def pdf_text(value) -> str:
    if value is None:
        return "N/D"
    text = str(value)
    replacements = {
        "↔": "<->",
        "→": "->",
        "—": "-",
        "–": "-",
        "²": "2",
        "“": '"',
        "”": '"',
        "’": "'",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return escape(text)


def number(value, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "N/D"


def fraction(value) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "N/D"
    return f"{numeric:.3f} ({numeric * 100:.1f}%)"



def nonnegative_interval(
    low,
    high,
    digits: int = 3,
) -> str:
    try:
        low_value = float(low)
        high_value = float(high)
    except (TypeError, ValueError):
        return "N/D"
    if not (
        math.isfinite(low_value)
        and math.isfinite(high_value)
    ):
        return "N/D"
    return (
        f"{max(0.0, low_value):.{digits}f} a "
        f"{high_value:.{digits}f} ppm"
    )


def operating_classification(
    score,
    threshold,
) -> str:
    try:
        score_value = float(score)
        threshold_value = float(threshold)
    except (TypeError, ValueError):
        return "N/D"
    if not (
        math.isfinite(score_value)
        and math.isfinite(threshold_value)
    ):
        return "N/D"
    return (
        "POSITIVA"
        if score_value >= threshold_value
        else "NEGATIVA"
    )
