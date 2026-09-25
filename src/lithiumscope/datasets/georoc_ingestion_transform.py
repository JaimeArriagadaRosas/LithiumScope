from __future__ import annotations

from pathlib import Path

import pandas as pd

from lithiumscope.datasets.georoc_ingestion_schema import (
    MAJOR_OXIDES,
)
from lithiumscope.tools.georoc_material_parser import (
    canonical_material_label,
    parse_georoc_material,
)


def quality_sum(
    frame: pd.DataFrame,
) -> pd.Series:
    if not all(
        column in frame.columns
        for column in MAJOR_OXIDES
    ):
        return pd.Series(
            pd.NA,
            index=frame.index,
            dtype="Float64",
        )
    numeric = frame[list(MAJOR_OXIDES)].apply(
        pd.to_numeric,
        errors="coerce",
    )
    complete = numeric.notna().all(axis=1)
    result = pd.Series(
        pd.NA,
        index=frame.index,
        dtype="Float64",
    )
    result.loc[complete] = numeric.loc[
        complete
    ].sum(axis=1)
    return result


def harmonize_chunk(
    chunk: pd.DataFrame,
    *,
    column_map: dict[str, str],
    source_name: str,
    source_file: str,
    allowed_material_types: tuple[str, ...],
    minimum_predictors: int,
) -> tuple[pd.DataFrame, int]:
    frame = chunk.rename(
        columns=column_map
    ).copy()

    if "Sample_type" in frame.columns:
        frame["Sample_type_raw"] = (
            frame["Sample_type"]
            .astype("string")
            .fillna("")
            .str.strip()
        )
        parsed_material = frame[
            "Sample_type_raw"
        ].map(
            parse_georoc_material
        )
        frame["Sample_material_code"] = (
            parsed_material.map(
                lambda item: item.code
            )
        )
        frame["Sample_material_batch_id"] = (
            parsed_material.map(
                lambda item: item.batch_id
            )
        )
        frame["Sample_type"] = (
            frame["Sample_material_code"].map(
                canonical_material_label
            )
        )

        if allowed_material_types:
            accepted_codes = {
                parse_georoc_material(
                    value
                ).code
                for value in allowed_material_types
            }
            if None in accepted_codes:
                raise ValueError(
                    "allowed_material_types contiene "
                    "un material GEOROC no reconocido."
                )
            frame = frame[
                frame["Sample_material_code"].isin(
                    accepted_codes
                )
            ].copy()

    numeric_columns = [
        column
        for column in (
            "Longitude", "Longitude_min", "Longitude_max",
            "Latitude", "Latitude_min", "Latitude_max",
            "Age (Ma)", *MAJOR_OXIDES, "Li_icpms",
            "Th_icpms", "U_icpms", "Rb_icpms",
            "Cs_icpms", "Nb_icpms", "Ta_icpms",
            "Pb_icpms", "Ba_icpms", "Sr_icpms",
            "Zr_icpms", "V_icpms", "Hf_icpms",
        )
        if column in frame.columns
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

    if (
        "Longitude" not in frame.columns
        and "Longitude_min" in frame.columns
    ):
        minimum = frame["Longitude_min"]
        maximum = frame.get(
            "Longitude_max",
            minimum,
        ).fillna(minimum)
        frame["Longitude"] = (
            minimum + maximum
        ) / 2.0

    if (
        "Latitude" not in frame.columns
        and "Latitude_min" in frame.columns
    ):
        minimum = frame["Latitude_min"]
        maximum = frame.get(
            "Latitude_max",
            minimum,
        ).fillna(minimum)
        frame["Latitude"] = (
            minimum + maximum
        ) / 2.0

    required = (
        "Li_icpms",
        "Longitude",
        "Latitude",
    )
    before = len(frame)
    if not all(
        column in frame.columns
        for column in required
    ):
        return frame.iloc[0:0].copy(), before

    frame = frame.dropna(
        subset=list(required)
    )
    frame = frame[
        frame["Longitude"].between(-180, 180)
        & frame["Latitude"].between(-90, 90)
    ].copy()

    predictor_columns = [
        column
        for column in (
            *MAJOR_OXIDES,
            "Th_icpms", "U_icpms", "Rb_icpms",
            "Cs_icpms", "Nb_icpms", "Ta_icpms",
            "Pb_icpms", "Ba_icpms", "Sr_icpms",
            "Zr_icpms", "V_icpms", "Hf_icpms",
        )
        if column in frame.columns
    ]
    if predictor_columns and minimum_predictors > 0:
        usable = frame[predictor_columns].notna().sum(
            axis=1
        )
        frame = frame[
            usable >= minimum_predictors
        ].copy()

    if "SUM (no water)" not in frame.columns:
        frame["SUM (no water)"] = quality_sum(
            frame
        )

    frame["source_dataset"] = source_name
    frame["source_file"] = source_file
    if "source_sample" not in frame.columns:
        frame["source_sample"] = [
            f"{Path(source_file).stem}_{index}"
            for index in frame.index
        ]

    return frame, before - len(frame)
