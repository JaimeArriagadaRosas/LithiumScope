from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from lithiumscope.tools.georoc_query_html import norm


@dataclass(frozen=True)
class DatasetHealth:
    rows: int
    columns: int
    li_nonempty: int
    valid_coordinate_pairs: int
    complete_li_coordinate_rows: int
    out_of_range_coordinates: int
    duplicate_rows: int

    def as_log_fields(self) -> dict[str, int]:
        return {
            "rows": self.rows,
            "columns": self.columns,
            "li_nonempty": self.li_nonempty,
            "valid_coordinate_pairs": self.valid_coordinate_pairs,
            "complete_li_coordinate_rows": self.complete_li_coordinate_rows,
            "out_of_range_coordinates": self.out_of_range_coordinates,
            "duplicate_rows": self.duplicate_rows,
        }


def _column_map(
    frame: pd.DataFrame,
) -> dict[str, str]:
    return {
        norm(column): str(column)
        for column in frame.columns
    }


def _pick_column(
    columns: dict[str, str],
    aliases: tuple[str, ...],
) -> str | None:
    wanted = tuple(
        norm(alias)
        for alias in aliases
    )
    for alias in wanted:
        if alias in columns:
            return columns[alias]
    for normalized, original in columns.items():
        if any(
            alias in normalized
            for alias in wanted
        ):
            return original
    return None


def assess_dataset_health(
    frame: pd.DataFrame,
) -> DatasetHealth:
    if frame.empty:
        raise RuntimeError(
            "La exportación GEOROC está vacía."
        )

    columns = _column_map(frame)
    li_column = _pick_column(
        columns,
        ("LI", "LI PPM"),
    )
    longitude_column = _pick_column(
        columns,
        (
            "LONGITUDE",
            "LONGITUDE MIN",
        ),
    )
    latitude_column = _pick_column(
        columns,
        (
            "LATITUDE",
            "LATITUDE MIN",
        ),
    )

    missing = [
        label
        for label, column in (
            ("LI", li_column),
            ("LONGITUDE", longitude_column),
            ("LATITUDE", latitude_column),
        )
        if column is None
    ]
    if missing:
        raise RuntimeError(
            "La exportación obtenida no corresponde "
            "al contrato LithiumScope. Faltan: "
            + ", ".join(missing)
        )

    li = frame[li_column]
    li_nonempty_mask = (
        li.notna()
        & li.astype(str).str.strip().ne("")
    )

    longitude = pd.to_numeric(
        frame[longitude_column],
        errors="coerce",
    )
    latitude = pd.to_numeric(
        frame[latitude_column],
        errors="coerce",
    )
    numeric_coordinates = (
        longitude.notna()
        & latitude.notna()
    )
    in_range = (
        longitude.between(-180, 180)
        & latitude.between(-90, 90)
    )
    valid_coordinates = (
        numeric_coordinates
        & in_range
    )
    complete = (
        li_nonempty_mask
        & valid_coordinates
    )

    li_nonempty = int(
        li_nonempty_mask.sum()
    )
    valid_coordinate_pairs = int(
        valid_coordinates.sum()
    )
    complete_rows = int(
        complete.sum()
    )
    out_of_range = int(
        (
            numeric_coordinates
            & ~in_range
        ).sum()
    )

    if li_nonempty == 0:
        raise RuntimeError(
            "La exportación GEOROC no contiene "
            "valores de litio utilizables."
        )
    if valid_coordinate_pairs == 0:
        raise RuntimeError(
            "La exportación GEOROC no contiene "
            "pares de coordenadas geográficas válidos."
        )
    if complete_rows == 0:
        raise RuntimeError(
            "La exportación GEOROC no contiene filas "
            "con litio y coordenadas válidas simultáneamente."
        )

    return DatasetHealth(
        rows=len(frame),
        columns=len(frame.columns),
        li_nonempty=li_nonempty,
        valid_coordinate_pairs=valid_coordinate_pairs,
        complete_li_coordinate_rows=complete_rows,
        out_of_range_coordinates=out_of_range,
        duplicate_rows=int(
            frame.duplicated().sum()
        ),
    )
