"""Evidence-gated catalog for independent geochemistry, without fabricating observations."""
from __future__ import annotations
import pandas as pd
import numpy as np

REQUIRED = ("sample_id","material","latitude","longitude","Li_value",
            "Li_unit","analytical_method","source","sample_date","independent_of_training")
COMPATIBLE_MATERIAL = {"rock", "whole_rock"}


def validate_geochemistry(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing = [col for col in REQUIRED if col not in frame]
    if missing:
        raise ValueError("Missing required evidence columns: " + ", ".join(missing))
    table = frame.copy()
    numeric = pd.to_numeric(table["Li_value"], errors="coerce")
    lat = pd.to_numeric(table["latitude"], errors="coerce")
    lon = pd.to_numeric(table["longitude"], errors="coerce")
    independent = table["independent_of_training"].astype(str).str.lower().isin(["true","yes","1"])
    compatible = (
        table["material"].astype(str).str.lower().isin(COMPATIBLE_MATERIAL)
        & table["Li_unit"].astype(str).str.lower().isin(["ppm", "mg/kg"])
        & numeric.ge(0) & np.isfinite(numeric)
        & lat.between(-90,90) & lon.between(-180,180)
        & independent
        & table["source"].notna() & table["analytical_method"].notna()
        & table["sample_id"].notna()
        & table["source"].astype(str).str.strip().ne("")
        & table["analytical_method"].astype(str).str.strip().ne("")
        & table["sample_id"].astype(str).str.strip().ne("")
    )
    table["evidence_status"] = compatible.map({True:"CANDIDATE_FOR_REVIEW",False:"NOT_ELIGIBLE"})
    return table.loc[compatible].copy(), table.loc[~compatible].copy()


