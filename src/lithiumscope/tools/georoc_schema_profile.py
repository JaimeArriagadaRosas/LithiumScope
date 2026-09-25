from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from lithiumscope.tools.georoc_query_html import norm


_CANDIDATE_TOKENS = (
    "MATERIAL",
    "SAMPLETYPE",
    "ROCK",
    "TYPE",
    "BATCH",
)


@dataclass(frozen=True)
class ColumnProfile:
    name: str
    nonempty: int
    unique_nonempty: int
    top_values: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class GeorocSchemaProfile:
    rows: int
    columns: tuple[str, ...]
    candidates: tuple[ColumnProfile, ...]


def _is_candidate(name: str) -> bool:
    normalized = norm(name)
    return any(
        token in normalized
        for token in _CANDIDATE_TOKENS
    )


def _profile_column(
    series: pd.Series,
    name: str,
    *,
    top_n: int,
) -> ColumnProfile:
    values = (
        series.astype("string")
        .fillna("")
        .str.strip()
    )
    nonempty = values[values.ne("")]
    counts = nonempty.value_counts(
        dropna=False
    )
    top_values = tuple(
        (
            str(value),
            int(count),
        )
        for value, count in counts.head(
            max(1, int(top_n))
        ).items()
    )
    return ColumnProfile(
        name=name,
        nonempty=int(len(nonempty)),
        unique_nonempty=int(
            nonempty.nunique(
                dropna=True
            )
        ),
        top_values=top_values,
    )


def profile_georoc_schema(
    frame: pd.DataFrame,
    *,
    max_candidates: int = 16,
    top_n: int = 12,
) -> GeorocSchemaProfile:
    columns = tuple(
        str(column)
        for column in frame.columns
    )
    candidates = tuple(
        _profile_column(
            frame[column],
            str(column),
            top_n=top_n,
        )
        for column in frame.columns
        if _is_candidate(str(column))
    )[: max(1, int(max_candidates))]

    return GeorocSchemaProfile(
        rows=len(frame),
        columns=columns,
        candidates=candidates,
    )


def log_georoc_schema_profile(
    run_log,
    profile: GeorocSchemaProfile,
) -> None:
    run_log.event(
        "schema_profile",
        "resumen",
        rows=profile.rows,
        columns=len(profile.columns),
        candidate_columns=len(
            profile.candidates
        ),
        column_names=", ".join(
            profile.columns
        ),
    )
    for candidate in profile.candidates:
        values = "; ".join(
            f"{value} [{count}]"
            for value, count
            in candidate.top_values
        )
        run_log.event(
            "schema_profile",
            "columna candidata",
            column=candidate.name,
            nonempty=candidate.nonempty,
            unique=candidate.unique_nonempty,
            top_values=values or "(sin valores)",
        )
