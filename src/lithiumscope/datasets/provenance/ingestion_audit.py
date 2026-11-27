"""GEOROC ingestion evidence serialization."""
from __future__ import annotations
import json
from pathlib import Path
from lithiumscope.core.hashing import file_sha256

def write_ingestion_audit(
    path: Path,
    *,
    source_name: str,
    files: tuple[Path, ...],
    rows_read: int,
    rows_kept: int,
    rejected: int,
    mapped_columns: dict[str, str],
    chunksize: int,
    allowed_material_types: tuple[str, ...],
    minimum_predictors: int,
    censoring: list[dict],
) -> None:
    audit = {
        "source": source_name,
        "source_sha256": {str(item): file_sha256(item) for item in files},
        "analytical_censoring": censoring,
        "rows_skipped_material_core_or_density": rows_read - rows_kept,
        "files": [
            str(item)
            for item in files
        ],
        "rows_read": rows_read,
        "rows_kept": rows_kept,
        "rows_rejected_missing_core_or_density": (
            rejected
        ),
        "mapped_columns": mapped_columns,
        "chunksize": chunksize,
        "allowed_material_types": list(
            allowed_material_types
        ),
        "minimum_predictors": minimum_predictors,
        "scientific_note": (
            "FE2O3T(WT%) is accepted as the total-iron "
            "counterpart for the current Fe2O3 predictor. "
            "The oxide-sum QC value is calculated only "
            "when all ten major oxides are available."
        ),
    }
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    path.write_text(
        json.dumps(
            audit,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


