from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


_COORDINATE_PAIRS = (
    ("Longitude", "Latitude"),
    ("Longitude (X)", "Latitude (Y)"),
    ("longitude", "latitude"),
)


def canonicalize_coordinates(
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


def concatenate_harmonized_sources(
    frames: Iterable[pd.DataFrame],
) -> pd.DataFrame:
    prepared = [
        canonicalize_coordinates(frame)
        for frame in frames
        if frame is not None and not frame.empty
    ]
    if not prepared:
        raise ValueError(
            "No harmonized datasets were supplied."
        )
    return pd.concat(
        prepared,
        ignore_index=True,
        sort=False,
    )


def deduplicate_harmonized_sources(
    frame: pd.DataFrame,
    *,
    coordinate_decimals: int = 5,
) -> tuple[pd.DataFrame, dict]:
    combined = canonicalize_coordinates(frame)
    rows_before = len(combined)

    dedupe_columns: list[str] = []
    identifier_candidates = [
        column
        for column in (
            "source_sample",
            "sample_id",
            "Sample",
        )
        if column in combined.columns
    ]
    if identifier_candidates:
        identifier = pd.Series(
            pd.NA,
            index=combined.index,
            dtype="string",
        )
        for column in identifier_candidates:
            values = (
                combined[column]
                .astype("string")
                .str.strip()
            )
            usable = values.notna() & values.ne("")
            identifier = identifier.mask(
                identifier.isna() & usable,
                values,
            )
        combined["_dedupe_sample_id"] = identifier
        dedupe_columns.append(
            "_dedupe_sample_id"
        )

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

    duplicate_mask = pd.Series(
        False,
        index=combined.index,
    )
    if dedupe_columns:
        duplicate_mask = combined.duplicated(
            subset=dedupe_columns,
            keep="first",
        )
        combined = combined.loc[
            ~duplicate_mask
        ].copy()

    combined = combined.drop(
        columns=[
            "_dedupe_sample_id",
            "_dedupe_longitude",
            "_dedupe_latitude",
        ],
        errors="ignore",
    ).reset_index(drop=True)

    audit = {
        "rows_before_deduplication": rows_before,
        "rows_after_deduplication": len(combined),
        "duplicates_removed": int(
            duplicate_mask.sum()
        ),
        "deduplication_columns": dedupe_columns,
        "coordinate_decimals": coordinate_decimals,
        "identifier_candidates": identifier_candidates,
        "rule": (
            "Conservative exact-key deduplication using a row-wise "
            "coalesced sample identifier (source_sample, sample_id "
            "or Sample) together with rounded coordinates. "
            "Same-location samples with different identifiers are retained."
        ),
    }
    return combined, audit


def merge_harmonized_sources(
    frames: Iterable[pd.DataFrame],
    *,
    coordinate_decimals: int = 5,
) -> tuple[pd.DataFrame, dict]:
    concatenated = concatenate_harmonized_sources(
        frames
    )
    merged, audit = deduplicate_harmonized_sources(
        concatenated,
        coordinate_decimals=coordinate_decimals,
    )
    audit["sources"] = (
        int(
            concatenated["source_dataset"].nunique(
                dropna=True
            )
        )
        if "source_dataset" in concatenated.columns
        else None
    )
    return merged, audit
