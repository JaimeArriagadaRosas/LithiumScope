from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from uuid import uuid4
from lithiumscope.core.hashing import file_sha256
from lithiumscope.datasets.readers.georoc_raw_reader import detect_georoc_encoding
from lithiumscope.datasets.readers.csv_reader import validate_csv_structure

import pandas as pd

from lithiumscope.core.logging.session import get_logger
from lithiumscope.datasets.provenance.ingestion_audit import write_ingestion_audit as _write_audit
from lithiumscope.datasets.adapters.georoc.schema import (
    GEOROC_ALIASES,
    MAJOR_OXIDES,
    infer_georoc_column_map, ingestion_layout,
)
from lithiumscope.datasets.adapters.georoc.transformation import (
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
    allow_incompatible_sources: bool = False,
) -> GeorocIngestionResult:
    if iron_total_mode not in {"original", "feot_equivalent"}:
        raise ValueError(f"Unsupported GEOROC iron_total_mode: {iron_total_mode}")
    paths = tuple(Path(path).resolve() for path in files)
    if len(set(paths)) != len(paths):
        raise ValueError("Duplicate GEOROC source paths")
    if not paths:
        raise ValueError(
            "No GEOROC CSV files were supplied."
        )

    if any(not path.is_file() for path in paths):
        raise FileNotFoundError("One or more declared GEOROC sources are missing")
    destination = Path(destination)
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
    audit_destination = Path(audit_destination)
    if destination.resolve() == audit_destination.resolve() or any(
            path in {destination.resolve(), audit_destination.resolve()} for path in paths):
        raise ValueError("GEOROC input and output paths must be distinct")
    if len({path.name for path in paths}) != len(paths):
        raise ValueError("GEOROC source basenames must be unique for row lineage")
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    audit_temporary = audit_destination.with_name(f".{audit_destination.name}.{uuid4().hex}.tmp")

    logger.info("GEOROC ingestion started: source=%s files=%d destination=%s", source_name, len(paths), destination)
    rows_read = 0
    rows_kept = 0
    rejected = 0
    mapped_union: dict[str, str] = {}
    wrote_header = False
    source_hashes = {str(path): file_sha256(path) for path in paths}
    encodings = {path: detect_georoc_encoding(path) for path in paths}
    headers = {path: validate_csv_structure(path, delimiter=",", encoding=encodings[path]) for path in paths}
    mappings, output_columns = ingestion_layout(paths, headers.__getitem__, iron_total_mode)
    censoring = []
    source_details = []

    try:
        for path in paths:
            logger.debug("GEOROC processing file=%s", path)
            mapping = mappings[path]
            mapped_union.update(mapping)

            missing = _validate_core_mapping(
                mapping
            )
            if allowed_material_types and "Sample_type" not in mapping.values():
                missing.append("Sample_type")
            if missing:
                if not allow_incompatible_sources:
                    raise ValueError(f"GEOROC source {path.name} lacks required columns: {missing}")
                source_details.append({"path": str(path), "state": "skipped_schema", "missing": missing,
                                       "encoding": encodings[path], "mapping": mapping})
                logger.warning(
                    "Skipping GEOROC file without core "
                    "columns %s: %s",
                    missing,
                    path,
                )
                continue

            first_read, first_kept = rows_read, rows_kept
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
                encoding=encodings[path],
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
            source_details.append({"path": str(path), "state": "processed", "encoding": encodings[path],
                                   "mapping": mapping, "rows_read": rows_read - first_read,
                                   "rows_kept": rows_kept - first_kept})
        if any(file_sha256(path) != source_hashes[str(path)] for path in paths):
            raise ValueError("GEOROC source changed during ingestion")
    except BaseException:
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

    try:
        _write_audit(
            audit_temporary,
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
        source_hashes=source_hashes,
        source_details=source_details,
        output_sha256=file_sha256(temporary),
        iron_total_mode=iron_total_mode,
        )
        from lithiumscope.datasets.provenance.publication import publish_pair
        publish_pair(temporary, destination, audit_temporary, audit_destination)
    finally:
        temporary.unlink(missing_ok=True)
        audit_temporary.unlink(missing_ok=True)

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
