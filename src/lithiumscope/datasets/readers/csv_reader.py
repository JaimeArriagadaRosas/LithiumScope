"""CSV decoding and strict row-width checks."""
import csv
from io import StringIO
from hashlib import sha256
from pathlib import Path
import pandas as pd
from lithiumscope.core.exceptions import InputValidationError
from lithiumscope.datasets.validation.tabular.headers import validate_headers as _validate_headers

CSV_ENCODINGS = ("utf-8-sig", "cp1252", "latin-1")
# The frozen public mirror has four blank, non-predictor headers.
_LEGACY_BLANK_HEADERS = {"8aa9c04aa5ad3837ae70bc0e9f9c1e37b4012f0e293e47c1ffe855ee13c56d7a": {28, 44, 81, 87}}


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


