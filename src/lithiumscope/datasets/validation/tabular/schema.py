"""Validate tabular header identity before parsing."""
from collections import Counter
import pandas as pd
from lithiumscope.core.exceptions import InputValidationError

def validate_headers(headers, *, allowed_blank: set[int] | None = None) -> list[str]:
    normalized = []
    allowed_blank = allowed_blank or set()
    for i, value in enumerate(headers):
        if pd.isna(value) or not str(value).strip():
            if i not in allowed_blank:
                raise InputValidationError(f"Empty header at column {i + 1}")
            value = f"Unnamed: {i}"
        normalized.append(str(value).strip().casefold())
    duplicate = [name for name, count in Counter(normalized).items() if count > 1]
    if duplicate:
        raise InputValidationError(f"Duplicate or ambiguous headers: {duplicate}")
    if len(normalized) < 2:
        raise InputValidationError("Expected a tabular structure with at least two columns")
    return normalized


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
