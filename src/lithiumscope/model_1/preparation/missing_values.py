from __future__ import annotations

import numpy as np
import pandas as pd

from lithiumscope.core.logging.session import get_logger

logger = get_logger("model_1.missing_values")

from lithiumscope.datasets.validation.missing_values import normalize_missing_values as normalize_missing_values


def find_age_column(frame: pd.DataFrame) -> str | None:
    for candidate in ("Age (Ma)", "Age (ma)", "Age", "age_ma"):
        if candidate in frame.columns:
            return candidate
    return None


def should_include_age(frame: pd.DataFrame, max_missing_fraction: float = 0.50) -> bool:
    candidate = find_age_column(frame)
    if candidate is None:
        return False
    numeric = pd.to_numeric(frame[candidate], errors="coerce")
    valid = numeric.notna() & np.isfinite(numeric) & (numeric >= 0)
    missing_fraction = float(1.0 - valid.mean())
    logger.info("Age column=%s missing_fraction=%.3f", candidate, missing_fraction)
    return missing_fraction <= max_missing_fraction
