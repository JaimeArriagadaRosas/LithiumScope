"""Concatenate, deduplicate and merge harmonized observations."""
from __future__ import annotations
from collections.abc import Iterable
import pandas as pd
from .coordinates import canonicalize_coordinates

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
    from lithiumscope.datasets.harmonization.deduplication import duplicate_evidence

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
