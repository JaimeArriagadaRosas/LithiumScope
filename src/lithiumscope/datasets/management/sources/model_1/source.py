from __future__ import annotations

from pathlib import Path


from lithiumscope.core.config import load_config
from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.logging.session import get_logger
from lithiumscope.core.paths import PROJECT_ROOT
from lithiumscope.datasets.adapters.georoc.ingestion import (
    ingest_georoc_files,
)
from lithiumscope.datasets.harmonization import (
    merge_harmonized_sources,
)
from lithiumscope.datasets.readers.tabular import (
    load_data,
)

from .configuration import project_path, verified_frozen_source
from .acquisition import ensure_georoc_files
from .assembly import assemble_sources

logger = get_logger("datasets.management.sources.model_1")


def _project_path(raw: str) -> Path:
    return project_path(raw, PROJECT_ROOT)


def prepare_model_1_training_source(
    base_dataset: Path,
    *,
    include_georoc: bool | None = None,
) -> Path:
    config = load_config("model_1")
    sources = config.get("data_sources", {})
    georoc = sources.get("georoc", {})

    use_georoc = (
        bool(georoc.get("enabled", False))
        if include_georoc is None
        else bool(include_georoc)
    )
    if not use_georoc:
        return base_dataset

    existing = _project_path(str(georoc.get('combined_path', 'data/processed/model_1/training_combined.csv')))
    expected = georoc.get('frozen_combined_sha256')
    if expected:
        verified_frozen_source(existing, expected, hash_file=file_sha256)
        logger.info('Reusing verified frozen M1 dataset: %s', existing)
        return existing

    raw_glob = str(
        georoc.get(
            "input_glob",
            "data/raw/model_1/georoc/*.csv",
        )
    )
    pattern = _project_path(raw_glob)
    georoc_files = sorted(
        pattern.parent.glob(pattern.name)
    )
    georoc_files = ensure_georoc_files(pattern, raw_glob, georoc_files)

    harmonized_path = _project_path(
        str(
            georoc.get(
                "harmonized_path",
                "data/interim/model_1/georoc_harmonized.csv",
            )
        )
    )
    merge_path = _project_path(
        str(
            georoc.get(
                "combined_path",
                "data/processed/model_1/training_combined.csv",
            )
        )
    )
    audit_path = _project_path(
        str(
            georoc.get(
                "audit_path",
                "data/processed/model_1/source_merge_audit.json",
            )
        )
    )

    ingestion = ingest_georoc_files(
        georoc_files,
        harmonized_path,
        chunksize=int(
            georoc.get("chunksize", 100_000)
        ),
        allowed_material_types=tuple(
            str(value)
            for value in georoc.get(
                "allowed_material_types",
                ["WHOLE ROCK"],
            )
        ),
        minimum_predictors=int(
            georoc.get("minimum_predictors", 8)
        ),
        iron_total_mode=str(georoc.get("iron_total_mode", "original")),
    )

    merged, merge_audit = assemble_sources(base_dataset, ingestion, merge_path, audit_path, georoc,
                                           load_data=load_data, merge_harmonized_sources=merge_harmonized_sources)

    logger.info(
        "Model 1 multi-source dataset ready: %s rows=%d "
        "georoc_rows=%d duplicates_removed=%d",
        merge_path,
        len(merged),
        ingestion.rows_kept,
        merge_audit["duplicates_removed"],
    )
    return merge_path
