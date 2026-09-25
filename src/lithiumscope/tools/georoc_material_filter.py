from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from lithiumscope.tools.georoc_query_html import norm


_MATERIAL_COLUMNS = (
    "TYPE OF MATERIAL",
    "SAMPLE TYPE",
    "MATERIAL",
)

_WHOLE_ROCK_VALUES = {
    "WHOLEROCK",
    "WR",
}


@dataclass(frozen=True)
class MaterialFilterResult:
    frame: pd.DataFrame
    material_column: str
    rows_before: int
    rows_after: int
    materials_seen: tuple[str, ...]

    @property
    def rows_removed(self) -> int:
        return self.rows_before - self.rows_after


def _find_material_column(
    frame: pd.DataFrame,
) -> str | None:
    by_normalized = {
        norm(column): str(column)
        for column in frame.columns
    }
    for candidate in _MATERIAL_COLUMNS:
        match = by_normalized.get(
            norm(candidate)
        )
        if match is not None:
            return match
    return None


def filter_whole_rock(
    frame: pd.DataFrame,
) -> MaterialFilterResult:
    material_column = _find_material_column(
        frame
    )
    if material_column is None:
        raise RuntimeError(
            "GEOROC no expuso una columna de material; "
            "no es posible demostrar el filtro WHOLE ROCK."
        )

    material = (
        frame[material_column]
        .astype("string")
        .fillna("")
        .str.strip()
    )
    normalized = material.map(norm)
    mask = normalized.isin(
        _WHOLE_ROCK_VALUES
    )
    filtered = frame.loc[mask].copy()

    materials_seen = tuple(
        sorted(
            {
                value
                for value in material.tolist()
                if value
            }
        )
    )
    if filtered.empty:
        preview = ", ".join(
            materials_seen[:12]
        )
        raise RuntimeError(
            "El filtro WHOLE ROCK dejó cero filas. "
            "Materiales observados: "
            + (preview or "(ninguno)")
        )

    return MaterialFilterResult(
        frame=filtered,
        material_column=material_column,
        rows_before=len(frame),
        rows_after=len(filtered),
        materials_seen=materials_seen,
    )
