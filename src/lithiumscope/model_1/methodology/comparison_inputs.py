"""Construct and harmonize experimental FeOT comparison cohorts."""
from __future__ import annotations

import numpy as np
import pandas as pd

from lithiumscope.datasets.adapters.georoc.schema import infer_georoc_column_map
from lithiumscope.datasets.adapters.georoc.transformation import harmonize_chunk
from lithiumscope.datasets.harmonization import canonicalize_coordinates
from lithiumscope.datasets.validation.chemistry.qc_protocols import protocol_masks
from lithiumscope.datasets.validation.tabular.missing_values import normalize_missing_values
from lithiumscope.model_1.preparation.detection_limits import clean_detection_limits


IRON_FEOT_TO_FE2O3T = 1.111348


def comparison_frame(raw: pd.DataFrame) -> pd.DataFrame:
    return canonicalize_coordinates(
        normalize_missing_values(clean_detection_limits(raw))
    )


def harmonize_feot_candidate(raw: pd.DataFrame, source_file: str) -> pd.DataFrame:
    """Convert total Fe as FeO to total Fe as Fe2O3 in an experimental copy.

    FEOT is TOTAL iron; never add it to any FE2O3 column. If FEOT is absent,
    the original Fe2O3 value can be used, but provenance is explicit.
    """
    copy = raw.copy(deep=True)
    feot = pd.to_numeric(copy.get("FEOT(WT%)"), errors="coerce")
    ferric = pd.to_numeric(copy.get("FE2O3(WT%)"), errors="coerce")
    if not isinstance(feot, pd.Series) or not isinstance(ferric, pd.Series):
        raise ValueError("Required raw Fe source columns are not present.")
    candidate = feot.mul(IRON_FEOT_TO_FE2O3T).where(feot.notna(), ferric)
    copy["FE2O3T(WT%)"] = candidate
    mapping = infer_georoc_column_map(copy.columns)
    if mapping.get("FE2O3T(WT%)") != "Fe2O3":
        raise RuntimeError("The GEOROC schema did not select the total iron column.")
    harmonized, _ = harmonize_chunk(
        copy,
        column_map=mapping,
        source_name="GEOROC",
        source_file=source_file,
        allowed_material_types=("WHOLE ROCK",),
        minimum_predictors=8,
    )
    harmonized["Fe_total_provenance"] = np.where(
        feot.reindex(harmonized.index).notna(),
        "FEOT_CONVERTED_TO_FE2O3T",
        "ORIGINAL_FE2O3",
    )
    return harmonized


def quality_candidate_frames(
    source: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Prepare three variants in-memory using the existing preprocessing rules."""
    frame = comparison_frame(source)
    frame["Li_icpms"] = pd.to_numeric(frame["Li_icpms"], errors="coerce")
    frame = frame.dropna(subset=["Li_icpms"]).copy()
    masks = protocol_masks(frame)
    return {
        name: frame.loc[masks[flag]].copy()
        for name, flag in (
            ("B", "B_reconstructed"),
            ("C", "C_experimental"),
        )
    }
