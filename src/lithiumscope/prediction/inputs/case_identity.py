"""Stable, collision-free identifiers for prediction cases."""
from __future__ import annotations
import pandas as pd


def ensure_case_ids(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    candidates = ("case_id", "Sample", "sample_id", "sample", "ID", "id")
    source = next((column for column in candidates if column in result.columns), None)
    if source is None:
        values = [f"case_{index + 1:04d}" for index in range(len(result))]
    else:
        values = [
            str(value).strip() if pd.notna(value) and str(value).strip()
            else f"case_{index + 1:04d}"
            for index, value in enumerate(result[source].tolist())
        ]

    reserved = set(values)
    used: set[str] = set()
    unique: list[str] = []
    for value in values:
        candidate = value
        count = 2
        while candidate in used:
            candidate = f"{value}__{count}"
            count += 1
            if candidate in reserved:
                continue
            if candidate not in used:
                break
        if candidate != value:
            while candidate in reserved or candidate in used:
                candidate = f"{value}__{count}"
                count += 1
        unique.append(candidate)
        used.add(candidate)
    result["case_id"] = unique
    return result
