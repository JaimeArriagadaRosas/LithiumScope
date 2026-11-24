"""Versioned analytical censoring; preserve raw observations in audit evidence."""
from __future__ import annotations

import re
import numpy as np
import pandas as pd

from lithiumscope.core.logging.session import get_logger

logger = get_logger("model_1.detection_limits")
_LEGACY = re.compile(r"^\s*([<>])\s*(-?\d+(?:\.\d+)?)\s*$")
_PATTERN = re.compile(r"^\s*(<=|>=|<|>|≤|≥)\s*([+-]?(?:\d+(?:[.,]\d*)?|[.,]\d+)(?:[eE][+-]?\d+)?)\s*$")
_MARKERS = {"bdl", "nd", "n.d.", "below detection limit", "not detected"}
_CHEMICAL = {"sio2", "tio2", "al2o3", "fe2o3", "feo", "mno", "mgo", "cao", "na2o",
             "k2o", "p2o5", "h2o+", "loi", "sum (no water)", "sum(no water)", "sum (water)"}


def analytical_columns(frame: pd.DataFrame) -> list[str]:
    return [name for name in frame.columns if str(name).strip().lower() in _CHEMICAL
            or str(name).lower().endswith(("_icpms", "_xrf"))]


def _legacy_convert(value):
    if not isinstance(value, str):
        return value
    match = _LEGACY.match(value)
    if match:
        symbol, raw = match.groups()
        number = float(raw)
        return number / 2.0 if symbol == "<" else number
    return value.strip()


def _convert(value, *, lower_fraction: float = 0.5, decimal_comma: bool = True):
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


def clean_detection_limits(frame: pd.DataFrame, *, columns: list[str] | None = None,
                           lower_fraction: float = 0.5, decimal_comma: bool = True,
                           version: int = 2, evidence: list[dict] | None = None) -> pd.DataFrame:
    cleaned = frame.copy()
    if version == 1:
        for name in cleaned:
            cleaned[name] = cleaned[name].map(_legacy_convert)
        return cleaned
    if version != 2 or not 0 <= lower_fraction <= 1:
        raise ValueError("Invalid censoring policy")
    selected = set(columns if columns is not None else analytical_columns(frame))
    counts = {}
    for name in cleaned:
        if name not in selected:
            continue
        values = []
        for idx, raw in cleaned[name].items():
            value, kind, limit = _convert(raw, lower_fraction=lower_fraction, decimal_comma=decimal_comma)
            values.append(value)
            if kind:
                counts[kind] = counts.get(kind, 0) + 1
                if evidence is not None:
                    source = frame.loc[idx, "source_dataset"] if "source_dataset" in frame else "unknown"
                    evidence.append({"row_index": str(idx), "source": str(source), "column": name,
                                     "original": str(raw), "censor_type": kind, "limit": limit,
                                     "substituted": value if pd.notna(value) else None})
        cleaned[name] = values
    logger.info("Analytical censoring version=%d counts=%s", version, counts)
    return cleaned
