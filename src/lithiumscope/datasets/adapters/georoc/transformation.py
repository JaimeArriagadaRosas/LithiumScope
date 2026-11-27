"""Orchestrate one harmonized GEOROC chunk, preserving scientific order."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from .chemistry import convert_chemistry, quality_sum
from .material_selection import select_material
from .row_selection import select_rows


def harmonize_chunk(
    chunk: pd.DataFrame,
    *,
    column_map: dict[str, str],
    source_name: str,
    source_file: str,
    allowed_material_types: tuple[str, ...],
    minimum_predictors: int,
    iron_total_mode: str = "original",
    censoring: list[dict] | None = None,
) -> tuple[pd.DataFrame, int]:
    if iron_total_mode not in {"original", "feot_equivalent"}:
        raise ValueError(f"Unsupported GEOROC iron_total_mode: {iron_total_mode}")
    before = len(chunk)
    frame = chunk.rename(
        columns=column_map
    ).copy()
    frame["source_row_position"] = chunk.index
    frame["source_li_unit"] = "ppm" if any(
        target == "Li_icpms" and "PPM" in raw.upper() for raw, target in column_map.items()
    ) else "unknown"
    frame["source_li_method"] = "unknown"
    frame["source_coordinate_crs"] = "unknown"
    if "Li_icpms" in frame:
        frame["source_li_raw"] = frame["Li_icpms"].astype("string")
    for name, source in (("source_feot_raw", "FeOT_total_as_FeO"), ("source_fe2o3_raw", "Fe2O3")):
        if source in frame:
            frame[name] = frame[source].astype("string")
    frame = convert_chemistry(frame, column_map, source_file, iron_total_mode, censoring)
    if "Sample_type" not in frame.columns and allowed_material_types:
        return frame.iloc[:0].copy(), before
    frame = select_material(frame, allowed_material_types)
    frame = select_rows(frame, minimum_predictors)
    if not {"Li_icpms", "Longitude", "Latitude"} <= set(frame.columns):
        return frame, before
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
