from __future__ import annotations

import pandas as pd

from lithiumscope.core.exceptions import InputValidationError
from lithiumscope.core.logger import get_logger

logger = get_logger("model_1.quality_control")


def _find_column(frame: pd.DataFrame, candidates: list[str]) -> str:
    normalized = {str(column).strip().lower(): column for column in frame.columns}
    for candidate in candidates:
        if candidate.strip().lower() in normalized:
            return normalized[candidate.strip().lower()]
    raise InputValidationError(
        "Quality-control column was not found. Tried: " + ", ".join(candidates)
    )


def apply_quality_control(
    frame: pd.DataFrame,
    candidates: list[str],
    minimum: float = 94.0,
    maximum: float = 102.0,
) -> pd.DataFrame:
    column = _find_column(frame, candidates)
    numeric = pd.to_numeric(frame[column], errors="coerce")
    mask = numeric.between(minimum, maximum, inclusive="both")
    result = frame.loc[mask].copy()
    logger.info(
        "Geochemical QC (%s in %.2f..%.2f): %d -> %d rows",
        column,
        minimum,
        maximum,
        len(frame),
        len(result),
    )
    return result
