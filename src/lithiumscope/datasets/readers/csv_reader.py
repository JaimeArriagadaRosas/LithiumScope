"""Stream encoding detection and strict CSV row-width validation."""
import csv
from pathlib import Path

import pandas as pd

from lithiumscope.core.exceptions import InputValidationError
from lithiumscope.core.hashing import file_sha256
from lithiumscope.datasets.validation.tabular.schema import validate_headers as _validate_headers

CSV_ENCODINGS = ("utf-8-sig", "cp1252", "latin-1")
# The frozen public mirror has four blank, non-predictor headers.
_LEGACY_BLANK_HEADERS = {"8aa9c04aa5ad3837ae70bc0e9f9c1e37b4012f0e293e47c1ffe855ee13c56d7a": {28, 44, 81, 87}}


def detect_text_encoding(path: Path, *, encoding: str | None = None) -> str:
    path = Path(path)
    errors = []
    if not path.is_file() or not path.stat().st_size:
        raise InputValidationError(f"Empty or missing text dataset: {path}")
    for codec in (encoding,) if encoding else CSV_ENCODINGS:
        try:
            with path.open("r", encoding=codec, newline="") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), ""):
                    if "\x00" in chunk:
                        raise InputValidationError("Unexpected null byte in text dataset")
            return codec
        except (UnicodeDecodeError, LookupError) as exc:
            errors.append(f"{codec}: {exc}")
    raise InputValidationError("No supported CSV encoding: " + " | ".join(errors))


def validate_csv_structure(path: Path, *, delimiter: str, encoding: str,
                           allowed_blank: set[int] | None = None) -> list[str]:
    if delimiter not in {",", ";", "\t"}:
        raise InputValidationError("Unsupported delimiter; declare comma, semicolon or tab")
    try:
        with Path(path).open("r", encoding=encoding, newline="") as handle:
            rows = csv.reader(handle, delimiter=delimiter, strict=True)
            header = next(rows, [])
            _validate_headers(header, allowed_blank=allowed_blank)
            for row in rows:
                if row and len(row) != len(header):
                    raise InputValidationError(f"Malformed row ending at line {rows.line_num}: expected {len(header)} fields, got {len(row)}")
    except csv.Error as exc:
        raise InputValidationError(f"Malformed CSV: {exc}") from exc
    return header


def _read_csv_with_fallback(path: Path, *, delimiter: str = ",", encoding: str | None = None) -> tuple[pd.DataFrame, str]:
    digest = file_sha256(path) if Path(path).is_file() else None
    codec = detect_text_encoding(path, encoding=encoding)
    validate_csv_structure(path, delimiter=delimiter, encoding=codec,
                           allowed_blank=_LEGACY_BLANK_HEADERS.get(digest))
    frame = pd.read_csv(path, sep=delimiter, encoding=codec)
    if file_sha256(path) != digest:
        raise InputValidationError("CSV changed during reading")
    return frame, codec
