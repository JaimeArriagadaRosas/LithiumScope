from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


_COORDINATE_PAIRS = (
    ("Longitude", "Latitude"),
    ("Longitude (X)", "Latitude (Y)"),
    ("longitude", "latitude"),
)


def _canonical_coordinates(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    result = frame.copy()
    pair = next(
        (
            pair
            for pair in _COORDINATE_PAIRS
            if pair[0] in result.columns
            and pair[1] in result.columns
        ),
        None,
    )
    if pair is None:
        return result

    longitude, latitude = pair
    result["Longitude"] = pd.to_numeric(
        result[longitude],
        errors="coerce",
    )
    result["Latitude"] = pd.to_numeric(
        result[latitude],
        errors="coerce",
    )
    return result


def merge_harmonized_sources(
    frames: Iterable[pd.DataFrame],
    *,
    coordinate_decimals: int = 5,
) -> tuple[pd.DataFrame, dict]:
    prepared = [
        _canonical_coordinates(frame)
        for frame in frames
        if frame is not None and not frame.empty
    ]
    if not prepared:
        raise ValueError(
            "No harmonized datasets were supplied."
        )

    combined = pd.concat(
        prepared,
        ignore_index=True,
        sort=False,
    )
    rows_before = len(combined)

    dedupe_columns: list[str] = []
    sample_id = next(
        (
            column
            for column in (
                "source_sample",
                "sample_id",
                "Sample",
            )
            if column in combined.columns
        ),
        None,
    )
    if sample_id is not None:
        dedupe_columns.append(sample_id)

    if {
        "Longitude",
        "Latitude",
    } <= set(combined.columns):
        combined["_dedupe_longitude"] = (
            pd.to_numeric(
                combined["Longitude"],
                errors="coerce",
            ).round(coordinate_decimals)
        )
        combined["_dedupe_latitude"] = (
            pd.to_numeric(
                combined["Latitude"],
                errors="coerce",
            ).round(coordinate_decimals)
        )
        dedupe_columns.extend(
            [
                "_dedupe_longitude",
                "_dedupe_latitude",
            ]
        )

    if dedupe_columns:
        combined = combined.drop_duplicates(
            subset=dedupe_columns,
            keep="first",
        )

    combined = combined.drop(
        columns=[
            "_dedupe_longitude",
            "_dedupe_latitude",
        ],
        errors="ignore",
    ).reset_index(drop=True)

    audit = {
        "sources": len(prepared),
        "rows_before_deduplication": rows_before,
        "rows_after_deduplication": len(combined),
        "duplicates_removed": (
            rows_before - len(combined)
        ),
        "deduplication_columns": dedupe_columns,
    }
    return combined, audit
