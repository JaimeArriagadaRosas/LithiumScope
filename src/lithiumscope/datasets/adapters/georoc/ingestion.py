from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from lithiumscope.core.logging.session import get_logger
from lithiumscope.datasets.provenance.ingestion_audit import write_ingestion_audit as _write_audit
from lithiumscope.datasets.adapters.georoc.ingestion_schema import (
    GEOROC_ALIASES,
    MAJOR_OXIDES,
    infer_georoc_column_map, ingestion_layout,
)
from lithiumscope.datasets.adapters.georoc.ingestion_transform import (
    harmonize_chunk,
)

logger = get_logger("datasets.georoc_ingestion")


@dataclass(frozen=True)
class GeorocIngestionResult:
    path: Path
    files: tuple[str, ...]
    rows_read: int
    rows_kept: int
    rows_rejected_missing_core: int
    mapped_columns: dict[str, str]
    audit_path: Path


def _read_header(path: Path) -> list[str]:
    return [
        str(column)
        for column in pd.read_csv(
            path,
            nrows=0,
            encoding="utf-8",
        ).columns
    ]


def _validate_core_mapping(
    mapping: dict[str, str],
) -> list[str]:
    produced = set(mapping.values())
    missing: list[str] = []
    if "Li_icpms" not in produced:
        missing.append("Li_icpms")
    if not {
        "Longitude",
        "Longitude_min",
    } & produced:
        missing.append("Longitude")
    if not {
        "Latitude",
        "Latitude_min",
    } & produced:
        missing.append("Latitude")
    return missing


def ingest_georoc_files(
    files: Iterable[Path],
    destination: Path,
    *,
    audit_path: Path | None = None,
    chunksize: int = 100_000,
    source_name: str = "GEOROC",
    allowed_material_types: tuple[str, ...] = (
        "WHOLE ROCK",
    ),
    minimum_predictors: int = 8,
    iron_total_mode: str = "original",
) -> GeorocIngestionResult:
    if iron_total_mode not in {"original", "feot_equivalent"}:
        raise ValueError(f"Unsupported GEOROC iron_total_mode: {iron_total_mode}")
    paths = tuple(
        Path(path)
        for path in files
        if Path(path).is_file()
    )
    if not paths:
        raise ValueError(
            "No GEOROC CSV files were supplied."
        )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    audit_destination = (
        audit_path
        or destination.with_suffix(
            ".audit.json"
        )
    )
    temporary = destination.with_suffix(
        destination.suffix + ".tmp"
    )
    temporary.unlink(missing_ok=True)

    logger.info("GEOROC ingestion started: source=%s files=%d destination=%s", source_name, len(paths), destination)
    rows_read = 0
    rows_kept = 0
    rejected = 0
    mapped_union: dict[str, str] = {}
    wrote_header = False
    mappings, output_columns = ingestion_layout(paths, _read_header, iron_total_mode)
    censoring = []

    try:
        for path in paths:
            logger.debug("GEOROC processing file=%s", path)
            mapping = mappings[path]
            mapped_union.update(mapping)

            missing = _validate_core_mapping(
                mapping
            )
            if missing:
                logger.warning(
                    "Skipping GEOROC file without core "
                    "columns %s: %s",
                    missing,
                    path,
                )
                continue

            usecols = list(mapping.keys())
            for chunk in pd.read_csv(
                path,
                usecols=usecols,
                chunksize=max(
                    1,
                    int(chunksize),
                ),
                low_memory=False,
                dtype=str,
                encoding="utf-8",
            ):
                rows_read += len(chunk)
                harmonized, dropped = (
                    harmonize_chunk(
                        chunk,
                        column_map=mapping,
                        source_name=source_name,
                        source_file=path.name,
                        allowed_material_types=(
                            allowed_material_types
                        ),
                        minimum_predictors=(
                            minimum_predictors
                        ),
                        iron_total_mode=iron_total_mode,
                        censoring=censoring,
                    )
                )
                rejected += dropped
                if harmonized.empty:
                    continue

                harmonized.reindex(columns=output_columns).to_csv(
                    temporary,
                    mode="a",
                    header=not wrote_header,
                    index=False,
                )
                wrote_header = True
                rows_kept += len(harmonized)
    except Exception:
        logger.exception("GEOROC ingestion failed: destination=%s", destination)
        temporary.unlink(
            missing_ok=True
        )
        raise

    if not wrote_header:
        temporary.unlink(
            missing_ok=True
        )
        raise ValueError(
            "GEOROC files produced no compatible rows."
        )

    temporary.replace(destination)
    _write_audit(
        audit_destination,
        source_name=source_name,
        files=paths,
        rows_read=rows_read,
        rows_kept=rows_kept,
        rejected=rejected,
        mapped_columns=mapped_union,
        chunksize=chunksize,
        allowed_material_types=(
            allowed_material_types
        ),
        minimum_predictors=minimum_predictors,
        censoring=censoring,
    )

    logger.info("GEOROC ingestion complete: rows_read=%d rows_kept=%d rejected=%d audit=%s", rows_read, rows_kept, rejected, audit_destination)
    return GeorocIngestionResult(
        path=destination,
        files=tuple(
            str(path)
            for path in paths
        ),
        rows_read=rows_read,
        rows_kept=rows_kept,
        rows_rejected_missing_core=(
            rejected
        ),
        mapped_columns=mapped_union,
        audit_path=audit_destination,
    )


__all__ = [
    "GEOROC_ALIASES",
    "MAJOR_OXIDES",
    "GeorocIngestionResult",
    "infer_georoc_column_map",
    "ingest_georoc_files",
]
