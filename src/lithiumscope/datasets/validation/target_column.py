from __future__ import annotations

import pandas as pd

from lithiumscope.core.exceptions import InputValidationError


def resolve_target(frame: pd.DataFrame, candidates: list[str]) -> str:
    normalized = {str(column).strip().lower(): column for column in frame.columns}
    for candidate in candidates:
        match = normalized.get(candidate.strip().lower())
        if match is not None:
            return match
    raise InputValidationError("Target column not found. Tried: " + ", ".join(candidates))
