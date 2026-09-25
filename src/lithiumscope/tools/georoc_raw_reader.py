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
) -> str:
    if (
        not path.is_file()
        or path.stat().st_size <= 0
    ):
        raise RuntimeError(
            "La descarga GEOROC está vacía."
        )

    for encoding in _ENCODINGS:
        try:
            with path.open(
                "r",
                encoding=encoding,
                newline="",
            ) as handle:
                for chunk in iter(
                    lambda: handle.read(1024 * 1024),
                    "",
                ):
                    if not chunk:
                        break
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
