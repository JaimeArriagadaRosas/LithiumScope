"""Public tabular reading API and source metadata."""
from __future__ import annotations
from hashlib import sha256
from pathlib import Path
import pandas as pd
from lithiumscope.core.exceptions import InputValidationError
from lithiumscope.core.logging.session import get_logger
from .csv_reader import CSV_ENCODINGS, _LEGACY_BLANK_HEADERS, _read_csv_with_fallback
from .excel_reader import read_excel
from lithiumscope.datasets.validation.tabular.headers import validate_headers as _validate_headers

logger = get_logger("datasets.readers.tabular")

def load_data(path: Path, *, quiet: bool = False, delimiter: str = ",",
              encoding: str | None = None, sheet_name: str | int = 0) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    metadata = {"format": suffix, "source_path": str(path)}
    if suffix == ".csv":
        frame, used_encoding = _read_csv_with_fallback(path, delimiter=delimiter, encoding=encoding)
        metadata.update(encoding=used_encoding, delimiter=delimiter,
                        legacy_blank_headers=sorted(_LEGACY_BLANK_HEADERS.get(sha256(path.read_bytes()).hexdigest(), set())))
        if not quiet:
            print(f"    ✓ CSV: {used_encoding}; delimitador={delimiter!r}")
    elif suffix in {".xlsx", ".xls"}:
        frame, sheet = read_excel(path, sheet_name)
        metadata.update(sheet=sheet)
    else:
        raise InputValidationError(f"Unsupported tabular format: {path.suffix}")
    if frame.empty:
        raise InputValidationError(f"Input dataset is empty: {path}")
    frame.attrs["tabular_source"] = metadata
    logger.info("Loaded tabular source %s with %d rows and %d columns", metadata, *frame.shape)
    if not quiet:
        print(f"    ✓ Dataset: {frame.shape[0]} filas × {frame.shape[1]} columnas")
    return frame
