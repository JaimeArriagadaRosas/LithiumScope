from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from lithiumscope.core.config import load_config
from lithiumscope.core.logging.session import get_logger
from lithiumscope.datasets.validation.concentration import valid_concentration
from lithiumscope.datasets.readers.tabular import load_data
from lithiumscope.datasets.validation.analytical import parse_analytical_value
from lithiumscope.datasets.harmonization.observations import reconcile_observations, stable_disambiguation
from lithiumscope.model_1.preparation.missing_values import normalize_missing_values
from lithiumscope.model_1.preparation.target_filtering import resolve_target

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

    frame = normalize_missing_values(load_data(dataset_path, quiet=True)).reset_index(drop=True)
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

    # For classification a censored assay is an interval, not a measured point.
    # Without a frozen endpoint it cannot safely be assigned to a class.
    parsed = [parse_analytical_value(v) for v in frame[target]]
    source = frame["source_dataset"] if "source_dataset" in frame else pd.Series(
        [dataset_path.name] * len(frame), index=frame.index)
    raw_ids = frame[sample_id_column] if sample_id_column else pd.Series(
        [""] * len(frame), index=frame.index)
    replicate_column = next((c for c in ("replicate_id", "analysis_id") if c in frame), None)
    replicate = frame[replicate_column] if replicate_column else pd.Series(
        [""] * len(frame), index=frame.index)
    output = pd.DataFrame(
        {
            "Li_icpms": pd.to_numeric(pd.Series([v if kind is None else np.nan for v, kind, _ in parsed]), errors="coerce"),
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

    numeric, target_mask, target_audit = valid_concentration(
        output["Li_icpms"], zero_policy=str(model_2["training"].get("target_zero_policy", "retain")),
    )
    output["Li_icpms"] = numeric
    output["_source"] = source.fillna("").astype(str)
    output["_original_id"] = raw_ids.fillna("").astype(str)
    output["_replicate"] = replicate.fillna("").astype(str)
    output["_raw_target"] = frame[target].astype(str)
    output["_censor"] = [kind or "" for _, kind, _ in parsed]
    output["_unit"] = frame["source_li_unit"].fillna("").astype(str) if "source_li_unit" in frame else ""
    output["_method"] = frame["source_li_method"].fillna("").astype(str) if "source_li_method" in frame else ""
    evidence = reconcile_observations(
        output, source="_source", sample="_original_id", replicate="_replicate",
        payload=["longitude", "latitude", "_raw_target", "_unit", "_method"])
    coordinates_ok = (output.longitude.between(-180, 180) & output.latitude.between(-90, 90))
    evidence["censor_type"] = output["_censor"]
    evidence["raw_target"] = output["_raw_target"]
    evidence["target_valid"] = target_mask
    evidence["coordinates_valid"] = coordinates_ok
    evidence["included"] = evidence.decision.eq("retain") & target_mask & coordinates_ok
    output = output.loc[evidence.included].copy()
    logger.info("M2 physical target validation: %s", target_audit)
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
    # Keep familiar IDs when unambiguous; resolve collisions using original identity.
    collisions = output.sample_id.duplicated(keep=False)
    for idx in output.index[collisions]:
        row = output.loc[idx]
        unknown = not row._source.strip() or not row._original_id.strip()
        digest = stable_disambiguation(row._source, row._original_id, row._replicate, idx if unknown else "")
        output.loc[idx, "sample_id"] = f"{row.sample_id}_{digest}"
    if output.sample_id.duplicated().any():
        raise ValueError("Unresolved sample ID collision")
    output = output[["Li_icpms", "longitude", "latitude", "sample_id", "spatial_group"]].reset_index(drop=True)
    output.attrs["source_reconciliation"] = evidence.to_dict("records")

    output.attrs["target_validation"] = target_audit
    logger.info(
        "Prepared %d georeferenced Li samples for Model 2 from %s",
        len(output),
        dataset_path,
    )
    return output
