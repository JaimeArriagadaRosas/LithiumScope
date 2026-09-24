from __future__ import annotations

import pandas as pd


def source_counts(frame: pd.DataFrame) -> pd.DataFrame:
    if "source_dataset" not in frame.columns:
        return pd.DataFrame(
            [{"source_dataset": "unknown", "rows": len(frame)}]
        )
    return (
        frame["source_dataset"]
        .fillna("unknown")
        .astype(str)
        .value_counts(dropna=False)
        .rename_axis("source_dataset")
        .reset_index(name="rows")
    )


def print_table(
    title: str,
    frame: pd.DataFrame,
    *,
    max_rows: int = 30,
) -> None:
    print(f"\n--- {title} ---")
    if frame.empty:
        print("(sin filas)")
        return
    print(frame.head(max_rows).to_string(index=False))
    if len(frame) > max_rows:
        print(f"... {len(frame) - max_rows} filas adicionales")


def dataset_summary(
    frame: pd.DataFrame,
    title: str,
) -> None:
    print(f"\n=== {title} ===")
    print(
        f"Filas: {len(frame):,} | "
        f"Columnas: {len(frame.columns):,} | "
        f"Celdas faltantes: {int(frame.isna().sum().sum()):,}"
    )

    print_table(
        "Filas por fuente",
        source_counts(frame),
    )

    missing = (
        frame.isna()
        .mean()
        .mul(100)
        .sort_values(ascending=False)
        .head(20)
        .rename("missing_percent")
        .reset_index(names="column")
    )
    print_table(
        "Top 20 columnas por porcentaje faltante",
        missing,
    )

    if "Li_icpms" not in frame.columns:
        return

    lithium = pd.to_numeric(
        frame["Li_icpms"],
        errors="coerce",
    )
    valid = lithium.dropna()
    if valid.empty:
        return

    stats = pd.DataFrame(
        [
            {
                "valid_li": int(valid.size),
                "missing_li": int(lithium.isna().sum()),
                "min": valid.min(),
                "q25": valid.quantile(0.25),
                "median": valid.median(),
                "mean": valid.mean(),
                "q75": valid.quantile(0.75),
                "max": valid.max(),
            }
        ]
    )
    print_table(
        "Distribución Li_icpms (ppm)",
        stats,
    )
