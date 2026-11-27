from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api.types import is_object_dtype, is_string_dtype

from lithiumscope.core.logging.session import get_logger

logger = get_logger("datasets.validation.tabular.missing_values")

_MISSING_TEXT = {"", "nan", "none", "null", "na", "n/a", "-"}


def normalize_missing_values(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.copy()
    for column in cleaned.columns:
        series = cleaned[column]
        if not (is_object_dtype(series.dtype) or is_string_dtype(series.dtype)):
            continue
        cleaned[column] = series.map(
            lambda value: np.nan
            if isinstance(value, str) and value.strip().lower() in _MISSING_TEXT
            else value
        )
    missing = int(cleaned.isna().sum().sum())
    logger.info("Missing-value normalization completed; total missing cells=%d", missing)
    return cleaned
