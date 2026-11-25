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
        numeric = float(value)
        return f"{numeric:.{digits}f}" if math.isfinite(numeric) else "N/D"
    except (TypeError, ValueError):
        return "N/D"


def fraction(value) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "N/D"
    return f"{numeric:.3f} ({numeric * 100:.1f}%)" if 0 <= numeric <= 1 else "N/D"



def interval_text(low, high, digits: int = 3) -> str:
    if number(low,digits)=="N/D" or number(high,digits)=="N/D":
        return "N/D"
    return f"{number(low,digits)} a {number(high,digits)} ppm"


# Legacy public names; the renderer never invokes these compatibility exports.
from lithiumscope.core.scientific.report_values import nonnegative_interval, operating_classification
