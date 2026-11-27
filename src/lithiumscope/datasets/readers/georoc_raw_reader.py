from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class GeorocRawRead:
    frame: pd.DataFrame
    encoding: str


def detect_georoc_encoding(path: Path) -> str:
    from lithiumscope.datasets.readers.csv_reader import detect_text_encoding
    from lithiumscope.core.exceptions import InputValidationError
    try:
        return detect_text_encoding(path)
    except InputValidationError as exc:
        raise RuntimeError(f"No se pudo determinar una codificación GEOROC compatible: {exc}") from exc


def read_georoc_raw(
    path: Path,
) -> GeorocRawRead:
    encoding = detect_georoc_encoding(path)
    try:
        frame = pd.read_csv(
            path,
            sep=",",
            encoding=encoding,
            low_memory=False,
        )
    except Exception as exc:
        raise RuntimeError(
            "No se pudo leer la exportación GEOROC "
            f"con encoding {encoding}: {exc}"
        ) from exc

    if frame.empty:
        raise RuntimeError(
            "La exportación GEOROC descargada está vacía."
        )

    return GeorocRawRead(
        frame=frame,
        encoding=encoding,
    )
