from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from lithiumscope.core.config import load_config
from lithiumscope.core.logger import get_logger
from lithiumscope.model_1.steps.step_01_load_data import load_data
from lithiumscope.model_1.steps.step_02_detection_limits import clean_detection_limits
from lithiumscope.model_1.steps.step_03_missing_values import normalize_missing_values
from lithiumscope.model_1.steps.step_05_target_filtering import resolve_target

logger = get_logger("model_2.sample_source")

LONGITUDE_CANDIDATES = (
    "Longitude (X)",
    "Logintude (X)",
    "Longitude",
    "longitude",
    "lon",
    "lng",
)
LATITUDE_CANDIDATES = (
    "Latitude (Y)",
    "Latitude",
    "latitude",
    "lat",
)
SAMPLE_ID_CANDIDATES = ("Sample", "sample_id", "Sample_ID", "ID", "id")


def _resolve_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str:
    normalized = {str(column).strip().lower(): column for column in frame.columns}
    for candidate in candidates:
        match = normalized.get(candidate.strip().lower())
        if match is not None:
            return str(match)
    raise ValueError(
        "No se encontró ninguna de las columnas requeridas: "
        + ", ".join(candidates)
    )


def _safe_sample_id(value, index: int) -> str:
    if pd.isna(value) or not str(value).strip():
        return f"sample_{index:05d}"
    text = str(value).strip()
    return "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in text
    )


def _spatial_group(latitude: float, longitude: float, degrees: float) -> str:
    lat_bin = int(np.floor((latitude + 90.0) / degrees))
    lon_bin = int(np.floor((longitude + 180.0) / degrees))
    return f"g_{lat_bin}_{lon_bin}"


def load_georeferenced_li_samples(dataset_path: Path) -> pd.DataFrame:
    config = load_config("model_1")
    model_2 = load_config("model_2")
    group_degrees = float(
        model_2["training"].get("spatial_group_degrees", 0.5)
    )

    frame = normalize_missing_values(
        clean_detection_limits(load_data(dataset_path, quiet=True))
    )
    target = resolve_target(frame, list(config["data"]["target_candidates"]))
    longitude = _resolve_column(frame, LONGITUDE_CANDIDATES)
    latitude = _resolve_column(frame, LATITUDE_CANDIDATES)

    sample_id_column = None
    normalized = {str(column).strip().lower(): column for column in frame.columns}
    for candidate in SAMPLE_ID_CANDIDATES:
        match = normalized.get(candidate.lower())
        if match is not None:
            sample_id_column = match
            break

    output = pd.DataFrame(
        {
            "Li_icpms": pd.to_numeric(frame[target], errors="coerce"),
            "longitude": pd.to_numeric(frame[longitude], errors="coerce"),
            "latitude": pd.to_numeric(frame[latitude], errors="coerce"),
        },
        index=frame.index,
    )
    if sample_id_column is None:
        output["sample_id"] = [
            _safe_sample_id(None, index)
            for index in range(len(output))
        ]
    else:
        output["sample_id"] = [
            _safe_sample_id(value, index)
            for index, value in enumerate(frame[sample_id_column].tolist())
        ]

    output = output.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["Li_icpms", "longitude", "latitude"]
    )
    output = output[
        output["longitude"].between(-180, 180)
        & output["latitude"].between(-90, 90)
    ].copy()
    output["spatial_group"] = [
        _spatial_group(lat, lon, group_degrees)
        for lat, lon in zip(
            output["latitude"],
            output["longitude"],
            strict=True,
        )
    ]
    output = output.drop_duplicates(
        subset=["sample_id", "longitude", "latitude"]
    ).reset_index(drop=True)

    logger.info(
        "Prepared %d georeferenced Li samples for Model 2 from %s",
        len(output),
        dataset_path,
    )
    return output
