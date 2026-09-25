from __future__ import annotations

from pathlib import Path

import pandas as pd

from lithiumscope.core.config import load_config
from lithiumscope.core.paths import DATA_DIR, PROJECT_ROOT
from lithiumscope.datasets.downloader import ensure_dataset
from lithiumscope.datasets.georoc_ingestion import ingest_georoc_files
from lithiumscope.datasets.harmonization import (
    concatenate_harmonized_sources,
    deduplicate_harmonized_sources,
)
from lithiumscope.model_1.steps.step_01_load_data import load_data
from lithiumscope.tools.pipeline_inspection_reporting import (
    dataset_summary,
    print_table,
)


def project_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else PROJECT_ROOT / path


def inspection_paths() -> dict[str, Path | str]:
    config = load_config("model_1")
    georoc = config.get("data_sources", {}).get("georoc", {})
    return {
        "input_glob": str(
            georoc.get(
                "input_glob",
                "data/raw/model_1/georoc/*.csv",
            )
        ),
        "harmonized": project_path(
            str(
                georoc.get(
                    "harmonized_path",
                    "data/interim/model_1/georoc_harmonized.csv",
                )
            )
        ),
        "concatenated": (
            DATA_DIR / "interim" / "model_1" / "training_concatenated.csv"
        ),
        "deduplicated": (
            DATA_DIR / "processed" / "model_1" / "training_combined.csv"
        ),
        "inspection_dir": (
            DATA_DIR / "processed" / "model_1" / "inspection"
        ),
    }


def georoc_files(raw_glob: str) -> list[Path]:
    pattern = project_path(raw_glob)
    return sorted(
        path
        for path in pattern.parent.glob(pattern.name)
        if path.is_file() and path.stat().st_size > 0
    )


def inspect_sources() -> dict:
    paths = inspection_paths()
    base_path = ensure_dataset("mamani09_public_mirror")
    base = load_data(base_path, quiet=True)
    files = georoc_files(str(paths["input_glob"]))

    print("\n=== STEP 1 · FUENTES BRUTAS ===")
    print(f"Mamani09: {base_path}")
    print(f"Mamani09 filas brutas: {len(base):,}")
    print(f"GEOROC archivos detectados: {len(files):,}")
    for path in files:
        size_mib = path.stat().st_size / 1048576
        print(f"  - {path.name} · {size_mib:.1f} MiB")

    return {
        "base_path": base_path,
        "base": base,
        "georoc_files": files,
        "paths": paths,
    }


def harmonize_georoc(state: dict | None = None) -> dict:
    state = state or inspect_sources()
    files = state["georoc_files"]
    if not files:
        raise RuntimeError("GEOROC no encontrado.")

    config = load_config("model_1")
    georoc = config.get("data_sources", {}).get("georoc", {})
    result = ingest_georoc_files(
        files,
        Path(state["paths"]["harmonized"]),
        chunksize=int(georoc.get("chunksize", 100_000)),
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
    frame = load_data(result.path, quiet=True)

    print("\n=== STEP 2 · GEOROC ARMONIZADO ===")
    print(f"Filas GEOROC leídas: {result.rows_read:,}")
    print(f"Filas GEOROC compatibles: {result.rows_kept:,}")
    print(
        "Filas descartadas en adaptación: "
        f"{result.rows_rejected_missing_core:,}"
    )
    print_table(
        "Mapeo de columnas GEOROC → LithiumScope",
        pd.DataFrame(
            [
                {
                    "georoc_column": source,
                    "lithiumscope_column": target,
                }
                for source, target in sorted(
                    result.mapped_columns.items()
                )
            ]
        ),
        max_rows=100,
    )
    dataset_summary(frame, "GEOROC armonizado")
    print(f"Auditoría: {result.audit_path}")

    state["georoc_result"] = result
    state["georoc"] = frame
    return state


def concatenate_sources(state: dict | None = None) -> dict:
    state = state or harmonize_georoc()
    if "georoc" not in state:
        state = harmonize_georoc(state)

    base = state["base"].copy()
    if "source_dataset" not in base.columns:
        base["source_dataset"] = "Mamani09 bootstrap"

    combined = concatenate_harmonized_sources(
        [base, state["georoc"]]
    )
    destination = Path(state["paths"]["concatenated"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(destination, index=False)

    print("\n=== STEP 3 · CONCATENACIÓN ===")
    dataset_summary(
        combined,
        "Mamani09 + GEOROC antes de deduplicar",
    )
    print(f"Guardado: {destination}")

    state["concatenated"] = combined
    state["concatenated_path"] = destination
    return state


def deduplicate_sources(state: dict | None = None) -> dict:
    state = state or concatenate_sources()
    if "concatenated" not in state:
        state = concatenate_sources(state)

    deduplicated, audit = deduplicate_harmonized_sources(
        state["concatenated"]
    )
    destination = Path(state["paths"]["deduplicated"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    deduplicated.to_csv(destination, index=False)

    print("\n=== STEP 4 · DEDUPLICACIÓN ===")
    print_table(
        "Auditoría de duplicados",
        pd.DataFrame([audit]),
    )
    dataset_summary(
        deduplicated,
        "Dataset combinado deduplicado",
    )
    print(f"Guardado: {destination}")

    state["deduplicated"] = deduplicated
    state["deduplicated_path"] = destination
    state["dedupe_audit"] = audit
    return state
