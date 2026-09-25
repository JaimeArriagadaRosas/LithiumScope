from __future__ import annotations

import pandas as pd


def normalize_georoc_frame(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    if frame.empty:
        raise RuntimeError(
            "No se puede normalizar un dataset GEOROC vacío."
        )

    result = frame.copy()
    result.columns = [
        " ".join(
            str(column)
            .replace("\u00a0", " ")
            .split()
        )
        for column in result.columns
    ]

    duplicated = result.columns.duplicated()
    if duplicated.any():
        names = sorted(
            {
                str(name)
                for name in result.columns[duplicated]
            }
        )
        raise RuntimeError(
            "La normalización GEOROC produjo "
            "columnas duplicadas: "
            + ", ".join(names)
        )

    object_columns = result.select_dtypes(
        include=["object", "string"]
    ).columns
    for column in object_columns:
        result[column] = result[column].map(
            lambda value: (
                value.replace("\u00a0", " ")
                if isinstance(value, str)
                else value
            )
        )

    return result
