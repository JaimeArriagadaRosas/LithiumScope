from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


_LONGITUDE_CANDIDATES = (
    "Longitude",
    "Longitude (X)",
    "Logintude (X)",
    "longitude",
)
_LATITUDE_CANDIDATES = (
    "Latitude",
    "Latitude (Y)",
    "latitude",
)


def _coalesce_numeric(
    frame: pd.DataFrame,
    candidates: tuple[str, ...],
) -> pd.Series | None:
    available = [
        column
        for column in candidates
        if column in frame.columns
    ]
    if not available:
        return None

    combined = pd.Series(
        float("nan"),
        index=frame.index,
        dtype=float,
    )
    for column in available:
        values = pd.to_numeric(
            frame[column],
            errors="coerce",
        )
        combined = combined.where(
            combined.notna(),
            values,
        )
    return combined


def canonicalize_coordinates(
    frame: pd.DataFrame,
    *, strict: bool = False,
) -> pd.DataFrame:
    """Legacy coalescence or strict new coordinate contract."""
    if strict:
        from lithiumscope.datasets.validation.coordinates import assess_coordinate_aliases
        return assess_coordinate_aliases(frame)[0]
    result = frame.copy()
    longitude = _coalesce_numeric(
        result,
        _LONGITUDE_CANDIDATES,
    )
    latitude = _coalesce_numeric(
        result,
        _LATITUDE_CANDIDATES,
    )
    if longitude is not None:
        result["Longitude"] = longitude
    if latitude is not None:
        result["Latitude"] = latitude
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
    """Remove only identical measurements with complete, unambiguous identities.

    Rounded coordinates identify candidates, not evidence of duplication.
    Conflicts and incomplete measurements remain available for scientific review.
    """
    from lithiumscope.datasets.harmonization_deduplication import duplicate_evidence

    combined = canonicalize_coordinates(frame, strict=True).reset_index(drop=True)
    keep, audit = duplicate_evidence(combined, coordinate_decimals=coordinate_decimals)
    return combined.loc[keep].reset_index(drop=True), audit


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
