from __future__ import annotations

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
