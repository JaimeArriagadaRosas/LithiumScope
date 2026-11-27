"""Analytical conversion, censoring evidence and oxide totals."""
from __future__ import annotations
import numpy as np
import pandas as pd
from lithiumscope.datasets.validation.analytical import parse_analytical_value
from .schema import MAJOR_OXIDES

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
    complete = (np.isfinite(numeric) & numeric.ge(0)).all(axis=1)
    result = pd.Series(
        pd.NA,
        index=frame.index,
        dtype="Float64",
    )
    result.loc[complete] = numeric.loc[
        complete
    ].sum(axis=1)
    return result


def convert_chemistry(frame, column_map, source_file, iron_total_mode, censoring):
    for raw_column, target in column_map.items():
        if target not in frame or not (target in MAJOR_OXIDES or target.endswith("_icpms") or target == "FeOT_total_as_FeO"):
            continue
        values = []
        for index, raw in frame[target].items():
            value, kind, limit = parse_analytical_value(raw, decimal_comma=False)
            values.append(value)
            if kind and censoring is not None:
                censoring.append({"source_file": source_file, "source_row_position": int(index),
                                  "column": raw_column, "canonical_column": target, "original": str(raw),
                                  "kind": kind, "limit": limit, "substituted": value if pd.notna(value) else None})
        frame[target] = values
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
        feot = feot.where(np.isfinite(feot) & feot.ge(0))
        original_fe = original_fe.where(np.isfinite(original_fe) & original_fe.ge(0))
        frame["Fe2O3"] = feot.mul(1.111348).where(feot.notna(), original_fe)
        frame["Fe_total_provenance"] = (
            "ORIGINAL_FE2O3"
        )
        frame.loc[feot.notna(), "Fe_total_provenance"] = (
            "FEOT_CONVERTED_TO_FE2O3T"
        )
        frame.loc[frame["Fe2O3"].isna(), "Fe_total_provenance"] = "UNAVAILABLE"
        frame = frame.drop(columns=["FeOT_total_as_FeO"], errors="ignore")

    return frame
