from __future__ import annotations

import pandas as pd

from lithiumscope.core.exceptions import InputValidationError
from lithiumscope.core.logger import get_logger

logger = get_logger("model_1.target_filtering")


def resolve_target(frame: pd.DataFrame, candidates: list[str]) -> str:
    normalized = {str(column).strip().lower(): column for column in frame.columns}
    for candidate in candidates:
        match = normalized.get(candidate.strip().lower())
        if match is not None:
            return match
    raise InputValidationError("Target column not found. Tried: " + ", ".join(candidates))


def filter_target(
    frame: pd.DataFrame,
    candidates: list[str],
    lower_quantile: float = 0.025,
    upper_quantile: float = 0.975,
) -> tuple[pd.DataFrame, str]:
    target = resolve_target(frame, candidates)
    result = frame.copy()
    result[target] = pd.to_numeric(result[target], errors="coerce")

    before = len(result)
    result = result.dropna(subset=[target])
    valid_count = len(result)

    low = float(result[target].quantile(lower_quantile))
    high = float(result[target].quantile(upper_quantile))
    result = result[result[target].between(low, high, inclusive="both")].copy()

    logger.info(
        "Target filtering %s: initial=%d valid=%d q%.3f=%.4f q%.3f=%.4f final=%d",
        target,
        before,
        valid_count,
        lower_quantile,
        low,
        upper_quantile,
        high,
        len(result),
    )
    return result, target
