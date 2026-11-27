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

from .model_1_assembly import assemble_sources

logger = get_logger("datasets.management.sources.model_1")

def project_path(raw, root):
    path = Path(raw).expanduser()
    return (path if path.is_absolute() else Path(root) / path).resolve()


def verified_frozen_source(path, expected, *, hash_file=file_sha256):
    if not path.is_file():
        raise FileNotFoundError("Frozen M1 dataset unavailable; restore verified bytes or explicitly select a separate unfrozen candidate configuration")
    if hash_file(path) != expected:
        raise ValueError("Frozen combined M1 dataset hash mismatch; no automatic overwrite")
    return path


def ensure_georoc_files(pattern, raw_glob, georoc_files):
    logger = get_logger("datasets.management.sources.model_1.acquisition")
    if not georoc_files:
        # Reuse the existing filtered GEOROC acquisition path from the lab.
        # A missing or failed GEOROC source must never silently be skipped.
        from lithiumscope.tools.acquisition.georoc.filtered import (
            GEOROC_FILTERED_NAME,
            acquire_filtered_georoc,
            write_acquisition_contract,
        )

        logger.info("GEOROC missing at %s; acquiring filtered source", raw_glob)
        print("GEOROC no disponible localmente; iniciando descarga filtrada.")
        pattern.parent.mkdir(parents=True, exist_ok=True)
        write_acquisition_contract(
            pattern.parent / "lithiumscope_georoc_query_contract.json"
        )
        acquire_filtered_georoc(
            destination=pattern.parent / GEOROC_FILTERED_NAME
        )
        georoc_files = sorted(pattern.parent.glob(pattern.name))
        if not georoc_files:
            raise RuntimeError(
                "La descarga GEOROC no produjo un CSV utilizable en "
                f"{raw_glob}."
            )
        logger.info("GEOROC available for harmonization: %s", georoc_files)

    return georoc_files


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
        logger.info('Selected base M1 source: %s', base_dataset)
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
