from __future__ import annotations

import re

import pandas as pd

from lithiumscope.core.logger import get_logger

logger = get_logger("model_1.detection_limits")
_PATTERN = re.compile(r"^\s*([<>])\s*(-?\d+(?:\.\d+)?)\s*$")


def _convert(value):
    if not isinstance(value, str):
        return value, None

    match = _PATTERN.match(value)
    if match:
        symbol, raw = match.groups()
        number = float(raw)
        converted = number / 2.0 if symbol == "<" else number
        return converted, "below" if symbol == "<" else "above"

    stripped = value.strip()
    return stripped, "trimmed" if stripped != value else None


def clean_detection_limits(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.copy()
    below = 0
    above = 0
    trimmed = 0

    for column in cleaned.columns:
        converted_values = []
        for value in cleaned[column].tolist():
            converted, kind = _convert(value)
            converted_values.append(converted)
            if kind == "below":
                below += 1
            elif kind == "above":
                above += 1
            elif kind == "trimmed":
                trimmed += 1
        cleaned[column] = converted_values

    logger.info(
        "Detection-limit normalization completed; below=%d above=%d whitespace_trimmed=%d analytical_conversions=%d",
        below,
        above,
        trimmed,
        below + above,
    )
    return cleaned
