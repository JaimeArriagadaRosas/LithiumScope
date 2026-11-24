"""Deterministic analytical token parsing, shared by ingestion and preparation."""
import re
import numpy as np

_PATTERN = re.compile(r"^\s*(<=|>=|<|>|≤|≥)\s*([+-]?(?:\d+(?:[.,]\d*)?|[.,]\d+)(?:[eE][+-]?\d+)?)\s*$")
_MARKERS = {"bdl", "nd", "n.d.", "below detection limit", "not detected"}

def parse_analytical_value(value, *, lower_fraction: float = 0.5, decimal_comma: bool = True):
    if not isinstance(value, str):
        return value, None, None
    stripped = value.strip()
    if stripped.casefold() in _MARKERS:
        return np.nan, "unquantified", None
    match = _PATTERN.fullmatch(stripped)
    if match:
        symbol, raw = match.groups()
        if "," in raw and not decimal_comma:
            return np.nan, "ambiguous_decimal", None
        limit = float(raw.replace(",", "."))
        if not np.isfinite(limit) or limit < 0:
            return np.nan, "invalid_limit", None
        below = symbol in {"<", "<=", "≤"}
        return limit * lower_fraction if below else limit, "below" if below else "above", limit
    if stripped.startswith(("<", ">", "≤", "≥")):
        return np.nan, "malformed_limit", None
    return stripped, None, None
