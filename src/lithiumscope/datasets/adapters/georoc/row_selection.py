"""Numeric, spatial and predictor-density admission rules."""
import numpy as np
import pandas as pd
from .schema import MAJOR_OXIDES


def select_rows(frame, minimum_predictors, *, rejections=None):
    def keep(mask, reason):
        nonlocal frame
        mask = mask.fillna(False)
        if rejections is not None:
            rejections.update({int(index): reason for index in frame.index[~mask]})
        frame = frame.loc[mask].copy()

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
    if not all(column in frame.columns for column in required):
        if rejections is not None:
            rejections.update({int(index): "missing_core_columns" for index in frame.index})
        return frame.iloc[0:0].copy()

    keep(np.isfinite(frame["Li_icpms"]) & frame["Li_icpms"].ge(0), "invalid_lithium")
    keep(frame["Longitude"].between(-180, 180) & frame["Latitude"].between(-90, 90),
         "invalid_coordinates")

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
    if minimum_predictors > 0:
        usable = (np.isfinite(frame[predictor_columns]) & frame[predictor_columns].ge(0)).sum(
            axis=1
        )
        keep(usable >= minimum_predictors, "insufficient_predictors")

    return frame
