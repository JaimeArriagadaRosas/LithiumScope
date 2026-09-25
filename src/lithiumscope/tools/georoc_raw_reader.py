from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class GeorocRawRead:
    frame: pd.DataFrame
    encoding: str


_ENCODINGS = (
    "utf-8-sig",
    "cp1252",
    "latin-1",
)


def detect_georoc_encoding(
    path: Path,
    *,
    sample_bytes: int = 512 * 1024,
) -> str:
    raw = path.read_bytes()[:sample_bytes]
    if not raw:
        raise RuntimeError(
            "La descarga GEOROC está vacía."
        )

    for encoding in _ENCODINGS:
        try:
            raw.decode(encoding)
            return encoding
        except UnicodeDecodeError:
            continue

    raise RuntimeError(
        "No se pudo determinar una codificación "
        "de texto compatible para GEOROC."
    )


def read_georoc_raw(
    path: Path,
) -> GeorocRawRead:
    encoding = detect_georoc_encoding(path)
    try:
        frame = pd.read_csv(
            path,
            sep=None,
            engine="python",
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
