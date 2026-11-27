from __future__ import annotations

from pathlib import Path

import pandas as pd
from uuid import uuid4
from lithiumscope.core.logging.session import get_logger

logger = get_logger("reporting.workbook")


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
    temporary = path.with_name(f".{path.stem}.{uuid4().hex}.xlsx")
    try:
        with pd.ExcelWriter(
            temporary,
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
        temporary.replace(path)
        logger.info("Prediction workbook written: %s", path)
    except Exception:
        logger.exception("Prediction workbook failed: %s", path)
        raise
    finally:
        temporary.unlink(missing_ok=True)
    return path
