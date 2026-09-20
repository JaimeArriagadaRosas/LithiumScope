from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold


LONGITUDE_CANDIDATES = (
    "Longitude (X)",
    "Logintude (X)",
    "Longitude",
    "longitude",
)
LATITUDE_CANDIDATES = (
    "Latitude (Y)",
    "Latitude",
    "latitude",
)


def _resolve_column(
    frame: pd.DataFrame,
    candidates: tuple[str, ...],
) -> str | None:
    normalized = {
        str(column).strip().lower(): str(column)
        for column in frame.columns
    }
    for candidate in candidates:
        match = normalized.get(
            candidate.strip().lower()
        )
        if match is not None:
            return match
    return None


def build_spatial_groups(
    frame: pd.DataFrame,
    *,
    degrees: float = 0.5,
) -> pd.Series | None:
    longitude_name = _resolve_column(
        frame,
        LONGITUDE_CANDIDATES,
    )
    latitude_name = _resolve_column(
        frame,
        LATITUDE_CANDIDATES,
    )
    if longitude_name is None or latitude_name is None:
        return None

    longitude = pd.to_numeric(
        frame[longitude_name],
        errors="coerce",
    )
    latitude = pd.to_numeric(
        frame[latitude_name],
        errors="coerce",
    )
    valid = (
        longitude.between(-180, 180)
        & latitude.between(-90, 90)
    )
    if not bool(valid.all()):
        return None

    size = float(degrees)
    if size <= 0:
        raise ValueError(
            "spatial_group_degrees must be positive."
        )
    lat_bin = np.floor(
        (latitude.to_numpy(dtype=float) + 90.0)
        / size
    ).astype(int)
    lon_bin = np.floor(
        (longitude.to_numpy(dtype=float) + 180.0)
        / size
    ).astype(int)
    return pd.Series(
        [
            f"g_{lat}_{lon}"
            for lat, lon in zip(
                lat_bin,
                lon_bin,
                strict=True,
            )
        ],
        index=frame.index,
        dtype="string",
        name="spatial_group",
    )


def materialize_regression_splits(
    x,
    y,
    groups: pd.Series | None,
    folds: int,
    seed: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], str]:
    if (
        groups is not None
        and int(groups.nunique()) >= int(folds)
    ):
        splitter = GroupKFold(
            n_splits=int(folds)
        )
        splits = splitter.split(
            x,
            y,
            groups,
        )
        strategy = "group_kfold_spatial"
    else:
        splitter = KFold(
            n_splits=int(folds),
            shuffle=True,
            random_state=int(seed),
        )
        splits = splitter.split(
            x,
            y,
        )
        strategy = "kfold_random"

    return (
        [
            (
                np.asarray(train_idx, dtype=int),
                np.asarray(test_idx, dtype=int),
            )
            for train_idx, test_idx in splits
        ],
        strategy,
    )


def nested_cv(
    outer_folds: int = 5,
    inner_folds: int = 5,
    random_seed: int = 42,
) -> tuple[KFold, KFold]:
    """Legacy random splitter factory kept for API compatibility."""
    outer = KFold(
        n_splits=outer_folds,
        shuffle=True,
        random_state=random_seed,
    )
    inner = KFold(
        n_splits=inner_folds,
        shuffle=True,
        random_state=random_seed,
    )
    return outer, inner
