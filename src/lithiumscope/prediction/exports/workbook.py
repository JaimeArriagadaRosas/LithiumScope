from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_workbook(
    path: Path,
    *,
    model_1: pd.DataFrame,
    model_2: pd.DataFrame,
    paired: pd.DataFrame,
    correlations: pd.DataFrame,
    concordance: pd.DataFrame,
    training_vs_external: pd.DataFrame,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(
        path,
        engine="openpyxl",
    ) as writer:
        model_1.to_excel(
            writer,
            sheet_name="modelo_1",
            index=False,
        )
        model_2.to_excel(
            writer,
            sheet_name="modelo_2",
            index=False,
        )
        paired.to_excel(
            writer,
            sheet_name="casos_emparejados",
            index=False,
        )
        correlations.to_excel(
            writer,
            sheet_name="correlaciones",
            index=False,
        )
        concordance.to_excel(
            writer,
            sheet_name="concordancia",
            index=False,
        )
        training_vs_external.to_excel(
            writer,
            sheet_name="training_vs_external",
            index=False,
        )
    return path
