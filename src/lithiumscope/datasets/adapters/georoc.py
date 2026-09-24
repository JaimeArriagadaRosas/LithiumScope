from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class GeorocAdapterConfig:
    """Map a verified GEOROC export into the LithiumScope contract.

    Column names are deliberately configuration-driven because GEOROC exports
    can vary by compilation. The adapter does not guess scientific units.
    """

    column_map: dict[str, str]
    lithium_scale_to_ppm: float = 1.0
    source_name: str = "GEOROC"
    required_output_columns: tuple[str, ...] = (
        "Li_icpms",
        "Longitude",
        "Latitude",
    )
    passthrough_columns: tuple[str, ...] = ()
    metadata: dict[str, str] = field(
        default_factory=dict
    )


def harmonize_georoc_frame(
    frame: pd.DataFrame,
    config: GeorocAdapterConfig,
) -> pd.DataFrame:
    if frame.empty:
        raise ValueError("GEOROC input frame is empty.")

    missing_source = [
        source
        for source in config.column_map
        if source not in frame.columns
    ]
    if missing_source:
        raise ValueError(
            "GEOROC export is missing mapped source columns: "
            + ", ".join(missing_source)
        )

    selected = list(config.column_map)
    selected.extend(
        column
        for column in config.passthrough_columns
        if column in frame.columns
        and column not in selected
    )
    result = frame[selected].copy().rename(
        columns=config.column_map
    )

    for column in (
        "Li_icpms",
        "Longitude",
        "Latitude",
    ):
        if column in result.columns:
            result[column] = pd.to_numeric(
                result[column],
                errors="coerce",
            )

    if "Li_icpms" in result.columns:
        result["Li_icpms"] = (
            result["Li_icpms"]
            * float(config.lithium_scale_to_ppm)
        )

    missing_output = [
        column
        for column in config.required_output_columns
        if column not in result.columns
    ]
    if missing_output:
        raise ValueError(
            "GEOROC mapping does not produce required columns: "
            + ", ".join(missing_output)
        )

    result["source_dataset"] = config.source_name
    for key, value in config.metadata.items():
        result[f"source_{key}"] = value

    return result
