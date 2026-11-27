from __future__ import annotations

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


