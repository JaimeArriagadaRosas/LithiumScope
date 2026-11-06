from __future__ import annotations

from pathlib import Path

import pandas as pd

from lithiumscope.datasets.georoc_ingestion_schema import (
    MAJOR_OXIDES,
)
from lithiumscope.tools.georoc.processing.georoc_material_parser import (
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
    iron_total_mode: str = "original",
) -> tuple[pd.DataFrame, int]:
    if iron_total_mode not in {"original", "feot_equivalent"}:
        raise ValueError(f"Unsupported GEOROC iron_total_mode: {iron_total_mode}")
    frame = chunk.rename(
        columns=column_map
    ).copy()
    if iron_total_mode == "feot_equivalent":
        # GEOROC FeOT is total iron expressed as FeO. Never add it to Fe2O3.
        # Prefer the FeOT total equivalent where reported; otherwise keep
        # the original Fe2O3/Fe2O3T supplied by the source.
        feot = pd.to_numeric(frame.get("FeOT_total_as_FeO"), errors="coerce")
        if not isinstance(feot, pd.Series):
            feot = pd.Series(float("nan"), index=frame.index)
        original_fe = pd.to_numeric(frame.get("Fe2O3"), errors="coerce")
        if not isinstance(original_fe, pd.Series):
            original_fe = pd.Series(float("nan"), index=frame.index)
        frame["Fe2O3"] = feot.mul(1.111348).where(feot.notna(), original_fe)
        frame["Fe_total_provenance"] = (
            "ORIGINAL_FE2O3"
        )
        frame.loc[feot.notna(), "Fe_total_provenance"] = (
            "FEOT_CONVERTED_TO_FE2O3T"
        )
        frame = frame.drop(columns=["FeOT_total_as_FeO"], errors="ignore")

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
