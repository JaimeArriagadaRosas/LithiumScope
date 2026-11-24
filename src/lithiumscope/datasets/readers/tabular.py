"""Strict shared tabular reader; preserve source conventions explicitly."""
from __future__ import annotations

import csv
from collections import Counter
from io import StringIO
from hashlib import sha256
from pathlib import Path

import pandas as pd

from lithiumscope.core.exceptions import InputValidationError
from lithiumscope.core.logging.session import get_logger

logger = get_logger("datasets.readers.tabular")
CSV_ENCODINGS = ("utf-8-sig", "cp1252", "latin-1")
# The frozen public mirror has four blank, non-predictor headers.
_LEGACY_BLANK_HEADERS = {"8aa9c04aa5ad3837ae70bc0e9f9c1e37b4012f0e293e47c1ffe855ee13c56d7a": {28, 44, 81, 87}}


def _validate_headers(headers, *, allowed_blank: set[int] | None = None) -> list[str]:
    normalized = []
    allowed_blank = allowed_blank or set()
    for i, value in enumerate(headers):
        if pd.isna(value) or not str(value).strip():
            if i not in allowed_blank:
                raise InputValidationError(f"Empty header at column {i + 1}")
            value = f"Unnamed: {i}"
        normalized.append(str(value).strip().casefold())
    duplicate = [name for name, count in Counter(normalized).items() if count > 1]
    if duplicate:
        raise InputValidationError(f"Duplicate or ambiguous headers: {duplicate}")
    if len(normalized) < 2:
        raise InputValidationError("Expected a tabular structure with at least two columns")
    return normalized


def _read_csv_with_fallback(path: Path, *, delimiter: str = ",", encoding: str | None = None) -> tuple[pd.DataFrame, str]:
    if delimiter not in {",", ";", "\t"}:
        raise InputValidationError("Unsupported delimiter; declare comma, semicolon or tab")
    errors = []
    raw = path.read_bytes()
    for codec in (encoding,) if encoding else CSV_ENCODINGS:
        try:
            content = raw.decode(codec)
        except UnicodeDecodeError as exc:
            errors.append(f"{codec}: {exc}")
            continue
        if "\x00" in content:
            raise InputValidationError("Unexpected null byte in text dataset")
        try:
            rows = csv.reader(StringIO(content), delimiter=delimiter, strict=True)
            header = next(rows, [])
            _validate_headers(header, allowed_blank=_LEGACY_BLANK_HEADERS.get(sha256(raw).hexdigest()))
            for row in rows:
                if row and len(row) != len(header):
                    raise InputValidationError(f"Malformed row ending at line {rows.line_num}: expected {len(header)} fields, got {len(row)}")
        except csv.Error as exc:
            raise InputValidationError(f"Malformed CSV: {exc}") from exc
        frame = pd.read_csv(StringIO(content), sep=delimiter)
        return frame, codec
    raise InputValidationError("No supported CSV encoding: " + " | ".join(errors))


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
        with pd.ExcelFile(path) as workbook:
            sheet = workbook.sheet_names[sheet_name] if isinstance(sheet_name, int) else sheet_name
            header = pd.read_excel(workbook, sheet_name=sheet, header=None, nrows=1)
            _validate_headers(header.iloc[0].tolist() if not header.empty else [])
            frame = pd.read_excel(workbook, sheet_name=sheet)
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
