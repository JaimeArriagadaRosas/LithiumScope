from __future__ import annotations

import pandas as pd

from lithiumscope.core.exceptions import InputValidationError


def resolve_target(frame: pd.DataFrame, candidates: list[str]) -> str:
    normalized = {}
    for column in frame.columns:
        key = str(column).strip().casefold()
        if key in normalized:
            raise InputValidationError(f"Duplicate or ambiguous target header: {column}")
        normalized[key] = column
    for candidate in candidates:
        match = normalized.get(candidate.strip().casefold())
        if match is not None:
            return match
    raise InputValidationError("Target column not found. Tried: " + ", ".join(candidates))
