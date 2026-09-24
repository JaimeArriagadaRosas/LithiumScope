from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from lithiumscope.core.config import load_config
from lithiumscope.core.logger import get_logger
from lithiumscope.core.paths import PROJECT_ROOT
from lithiumscope.datasets.georoc_ingestion import (
    ingest_georoc_files,
)
from lithiumscope.datasets.harmonization import (
    merge_harmonized_sources,
)
from lithiumscope.model_1.steps.step_01_load_data import (
    load_data,
)

logger = get_logger("datasets.model_1_sources")


def _project_path(raw: str) -> Path:
    path = Path(raw)
    return (
        path
        if path.is_absolute()
        else PROJECT_ROOT / path
    )


def _source_name(frame: pd.DataFrame, fallback: str) -> pd.DataFrame:
    result = frame.copy()
    if "source_dataset" not in result.columns:
        result["source_dataset"] = fallback
    return result


def prepare_model_1_training_source(
    base_dataset: Path,
) -> Path:
    config = load_config("model_1")
    sources = config.get("data_sources", {})
    georoc = sources.get("georoc", {})

    if not bool(georoc.get("enabled", False)):
        return base_dataset

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
    if not georoc_files:
        logger.warning(
            "GEOROC integration enabled but no local CSV files "
            "matched %s; using base dataset only.",
            raw_glob,
        )
        return base_dataset

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
    )

    base = _source_name(
        load_data(base_dataset, quiet=True),
        "Mamani09 bootstrap",
    )
    external = load_data(
        ingestion.path,
        quiet=True,
    )
    merged, merge_audit = merge_harmonized_sources(
        [base, external]
    )

    merge_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    merged.to_csv(
        merge_path,
        index=False,
    )

    audit = {
        "base_dataset": str(base_dataset),
        "georoc_files": list(ingestion.files),
        "georoc_ingestion": {
            "rows_read": ingestion.rows_read,
            "rows_kept": ingestion.rows_kept,
            "audit_path": str(ingestion.audit_path),
        },
        "merge": merge_audit,
        "combined_path": str(merge_path),
    }
    audit_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    audit_path.write_text(
        json.dumps(
            audit,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    logger.info(
        "Model 1 multi-source dataset ready: %s rows=%d "
        "georoc_rows=%d duplicates_removed=%d",
        merge_path,
        len(merged),
        ingestion.rows_kept,
        merge_audit["duplicates_removed"],
    )
    return merge_path
