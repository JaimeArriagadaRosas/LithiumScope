from __future__ import annotations

import numpy as np
import pandas as pd

from lithiumscope.core.logger import get_logger

logger = get_logger("model_1.missing_values")

_MISSING_TEXT = {"", "nan", "none", "null", "na", "n/a", "-"}


def normalize_missing_values(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.copy()
    for column in cleaned.select_dtypes(include=["object"]).columns:
        cleaned[column] = cleaned[column].map(
            lambda value: np.nan
            if isinstance(value, str) and value.strip().lower() in _MISSING_TEXT
            else value
        )
    missing = int(cleaned.isna().sum().sum())
    logger.info("Missing-value normalization completed; total missing cells=%d", missing)
    return cleaned


def find_age_column(frame: pd.DataFrame) -> str | None:
    for candidate in ("Age (Ma)", "Age (ma)", "Age", "age_ma"):
        if candidate in frame.columns:
            return candidate
    return None


def should_include_age(frame: pd.DataFrame, max_missing_fraction: float = 0.50) -> bool:
    candidate = find_age_column(frame)
    if candidate is None:
        return False
    missing_fraction = float(frame[candidate].isna().mean())
    logger.info("Age column=%s missing_fraction=%.3f", candidate, missing_fraction)
    return missing_fraction <= max_missing_fraction
