from __future__ import annotations

from pathlib import Path

import pandas as pd

from lithiumscope.core.exceptions import InputValidationError
from lithiumscope.core.logger import get_logger

logger = get_logger("model_1.load_data")

CSV_ENCODINGS = ("utf-8", "utf-8-sig", "cp1252", "latin-1")


def _read_csv_with_fallback(path: Path) -> tuple[pd.DataFrame, str]:
    errors: list[str] = []
    for encoding in CSV_ENCODINGS:
        try:
            frame = pd.read_csv(path, encoding=encoding)
            return frame, encoding
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")
    raise InputValidationError(
        "No se pudo decodificar el CSV con las codificaciones soportadas. "
        + " | ".join(errors)
    )


def load_data(path: Path, *, quiet: bool = False) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        frame, encoding = _read_csv_with_fallback(path)
        logger.info("CSV encoding selected: %s", encoding)
        if not quiet:
            print(f"    ✓ CSV cargado con codificación: {encoding}")
    elif suffix in {".xlsx", ".xls"}:
        frame = pd.read_excel(path)
    else:
        raise InputValidationError(f"Unsupported tabular format: {path.suffix}")

    if frame.empty:
        raise InputValidationError(f"Input dataset is empty: {path}")

    logger.info("Loaded dataset %s with %d rows and %d columns", path, *frame.shape)
    if not quiet:
        print(f"    ✓ Dataset: {frame.shape[0]} filas × {frame.shape[1]} columnas")
    return frame
