from __future__ import annotations

from pathlib import Path

import pandas as pd

from lithiumscope.core.exceptions import InputValidationError
from lithiumscope.core.logger import get_logger

logger = get_logger("model_1.load_data")


def load_data(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(path)
    elif suffix in {".xlsx", ".xls"}:
        frame = pd.read_excel(path)
    else:
        raise InputValidationError(f"Unsupported tabular format: {path.suffix}")

    if frame.empty:
        raise InputValidationError(f"Input dataset is empty: {path}")

    logger.info("Loaded dataset %s with %d rows and %d columns", path, *frame.shape)
    return frame
