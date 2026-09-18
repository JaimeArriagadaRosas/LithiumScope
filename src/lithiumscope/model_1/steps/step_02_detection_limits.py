from __future__ import annotations

import re

import pandas as pd

from lithiumscope.core.logger import get_logger

logger = get_logger("model_1.detection_limits")
_PATTERN = re.compile(r"^\s*([<>])\s*(-?\d+(?:\.\d+)?)\s*$")


def _convert(value):
    if not isinstance(value, str):
        return value
    match = _PATTERN.match(value)
    if not match:
        return value.strip()
    symbol, raw = match.groups()
    number = float(raw)
    return number / 2.0 if symbol == "<" else number


def clean_detection_limits(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.copy()
    conversions = 0
    for column in cleaned.columns:
        if cleaned[column].dtype != object:
            continue
        original = cleaned[column].copy()
        cleaned[column] = cleaned[column].map(_convert)
        conversions += int((original.astype(str) != cleaned[column].astype(str)).sum())
    logger.info("Detection-limit normalization completed; changed cells=%d", conversions)
    return cleaned
