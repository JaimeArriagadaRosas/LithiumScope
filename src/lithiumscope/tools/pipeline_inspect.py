from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from lithiumscope.cli.prompts import choose
from lithiumscope.core.config import load_config
from lithiumscope.core.paths import DATA_DIR, PROJECT_ROOT
from lithiumscope.datasets.downloader import ensure_dataset
from lithiumscope.datasets.georoc_ingestion import (
    ingest_georoc_files,
)
from lithiumscope.datasets.harmonization import (
    concatenate_harmonized_sources,
    deduplicate_harmonized_sources,
)
from lithiumscope.model_1.pipeline import (
    prepare_training_data,
)
from lithiumscope.model_1.steps.step_01_load_data import (
    load_data,
)
from lithiumscope.model_2.data.sample_source import (
    load_georeferenced_li_samples,
)


def _project_path(raw: str) -> Path:
    path = Path(raw)
    return (
        path
        if path.is_absolute()
        else PROJECT_ROOT / path
    )


def _paths() -> dict[str, Path | str]:
    config = load_config("model_1")
    georoc = config.get("data_sources", {}).get(
        "georoc",
        {},
    )
    return {
        "input_glob": str(
            georoc.get(
                "input_glob",
                "data/raw/model_1/georoc/*.csv",
            )
        ),
        "harmonized": _project_path(
            str(
                georoc.get(
                    "harmonized_path",
                    "data/interim/model_1/georoc_harmonized.csv",
                )
            )
        ),
        "concatenated": (
            DATA_DIR
            / "interim"
            / "model_1"
            / "training_concatenated.csv"
        ),
        "deduplicated": (
            DATA_DIR
            / "processed"
            / "model_1"
            / "training_combined.csv"
        ),
        "inspection_dir": (
            DATA_DIR
            / "processed"
            / "model_1"
            / "inspection"
        ),
    }


def _georoc_files(raw_glob: str) -> list[Path]:
    pattern = _project_path(raw_glob)
    return sorted(
        pattern.parent.glob(pattern.name)
    )


def _source_counts(frame: pd.DataFrame) -> pd.DataFrame:
    if "source_dataset" not in frame.columns:
        return pd.DataFrame(
            [{"source_dataset": "unknown", "rows": len(frame)}]
        )
    counts = (
        frame["source_dataset"]
        .fillna("unknown")
        .astype(str)
        .value_counts(dropna=False)
        .rename_axis("source_dataset")
        .reset_index(name="rows")
    )
    return counts


def _print_table(
    title: str,
    frame: pd.DataFrame,
    *,
    max_rows: int = 30,
) -> None:
    print(f"\n--- {title} ---")
    if frame.empty:
        print("(sin filas)")
        return
    print(
        frame.head(max_rows).to_string(
            index=False
        )
    )
    if len(frame) > max_rows:
        print(
            f"... {len(frame) - max_rows} filas adicionales"
        )


def _dataset_summary(
    frame: pd.DataFrame,
    title: str,
) -> None:
    print(f"\n=== {title} ===")
    print(
        f"Filas: {len(frame):,} | "
        f"Columnas: {len(frame.columns):,} | "
        f"Celdas faltantes: {int(frame.isna().sum().sum()):,}"
    )

    _print_table(
        "Filas por fuente",
        _source_counts(frame),
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
    _print_table(
        "Top 20 columnas por porcentaje faltante",
        missing,
    )

    if "Li_icpms" in frame.columns:
        li = pd.to_numeric(
            frame["Li_icpms"],
            errors="coerce",
        )
        valid = li.dropna()
        if not valid.empty:
            stats = pd.DataFrame(
                [
                    {
                        "valid_li": int(valid.size),
                        "missing_li": int(li.isna().sum()),
                        "min": valid.min(),
                        "q25": valid.quantile(0.25),
                        "median": valid.median(),
                        "mean": valid.mean(),
                        "q75": valid.quantile(0.75),
                        "max": valid.max(),
                    }
                ]
            )
            _print_table(
                "Distribución Li_icpms (ppm)",
                stats,
            )


def inspect_sources() -> dict:
    paths = _paths()
    base_path = ensure_dataset(
        "mamani09_public_mirror"
    )
    base = load_data(
        base_path,
        quiet=True,
    )
    files = _georoc_files(
        str(paths["input_glob"])
    )

    print("\n=== STEP 1 · FUENTES BRUTAS ===")
    print(f"Mamani09: {base_path}")
    print(
        f"Mamani09 filas brutas: {len(base):,}"
    )
    print(
        f"GEOROC archivos detectados: {len(files):,}"
    )
    for path in files:
        size_gib = path.stat().st_size / (1024 ** 3)
        print(
            f"  - {path.name} · {size_gib:.2f} GiB"
        )
    if not files:
        print(
            "GEOROC: no hay CSV en "
            f"{paths['input_glob']}"
        )

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
        raise RuntimeError(
            "No existen CSV GEOROC para armonizar."
        )

    config = load_config("model_1")
    georoc = config.get("data_sources", {}).get(
        "georoc",
        {},
    )
    result = ingest_georoc_files(
        files,
        Path(state["paths"]["harmonized"]),
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
            georoc.get(
                "minimum_predictors",
                8,
            )
        ),
    )
    frame = load_data(
        result.path,
        quiet=True,
    )

    print("\n=== STEP 2 · GEOROC ARMONIZADO ===")
    print(
        f"Filas GEOROC leídas: {result.rows_read:,}"
    )
    print(
        f"Filas GEOROC compatibles: {result.rows_kept:,}"
    )
    print(
        "Filas descartadas en adaptación: "
        f"{result.rows_rejected_missing_core:,}"
    )
    _dataset_summary(
        frame,
        "GEOROC armonizado",
    )
    print(
        f"Auditoría: {result.audit_path}"
    )

    state["georoc_result"] = result
    state["georoc"] = frame
    return state


def concatenate_sources(
    state: dict | None = None,
) -> dict:
    state = state or harmonize_georoc()
    if "georoc" not in state:
        state = harmonize_georoc(state)

    base = state["base"].copy()
    if "source_dataset" not in base.columns:
        base["source_dataset"] = (
            "Mamani09 bootstrap"
        )
    combined = concatenate_harmonized_sources(
        [base, state["georoc"]]
    )
    destination = Path(
        state["paths"]["concatenated"]
    )
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    combined.to_csv(
        destination,
        index=False,
    )

    print("\n=== STEP 3 · CONCATENACIÓN ===")
    _dataset_summary(
        combined,
        "Mamani09 + GEOROC antes de deduplicar",
    )
    print(f"Guardado: {destination}")

    state["concatenated"] = combined
    state["concatenated_path"] = destination
    return state


def deduplicate_sources(
    state: dict | None = None,
) -> dict:
    state = state or concatenate_sources()
    if "concatenated" not in state:
        state = concatenate_sources(state)

    deduplicated, audit = (
        deduplicate_harmonized_sources(
            state["concatenated"]
        )
    )
    destination = Path(
        state["paths"]["deduplicated"]
    )
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    deduplicated.to_csv(
        destination,
        index=False,
    )

    print("\n=== STEP 4 · DEDUPLICACIÓN ===")
    _print_table(
        "Auditoría de duplicados",
        pd.DataFrame([audit]),
    )
    _dataset_summary(
        deduplicated,
        "Dataset combinado deduplicado",
    )
    print(f"Guardado: {destination}")

    state["deduplicated"] = deduplicated
    state["deduplicated_path"] = destination
    state["dedupe_audit"] = audit
    return state


def inspect_model_1_pipeline(
    state: dict | None = None,
) -> dict:
    state = state or deduplicate_sources()
    if "deduplicated_path" not in state:
        state = deduplicate_sources(state)

    print(
        "\n=== STEP 5 · LIMPIEZA REAL DEL MODELO 1 ==="
    )
    prepared = prepare_training_data(
        Path(state["deduplicated_path"]),
        model_family="random_forest",
    )

    audit = prepared.audit.to_frame()
    _print_table(
        "Filas después de cada step M1",
        audit,
    )
    _dataset_summary(
        prepared.frame,
        "Dataset final que entra a M1",
    )

    schema_rows = pd.DataFrame(
        [
            {
                "type": "numeric",
                "count": len(
                    prepared.schema.numeric
                ),
                "columns": ", ".join(
                    prepared.schema.numeric
                ),
            },
            {
                "type": "categorical",
                "count": len(
                    prepared.schema.categorical
                ),
                "columns": ", ".join(
                    prepared.schema.categorical
                ),
            },
        ]
    )
    _print_table(
        "Esquema final de features M1",
        schema_rows,
    )

    inspection_dir = Path(
        state["paths"]["inspection_dir"]
    )
    inspection_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    audit_path = (
        inspection_dir
        / "model_1_pipeline_audit.csv"
    )
    final_path = (
        inspection_dir
        / "model_1_final_samples.csv"
    )
    audit.to_csv(
        audit_path,
        index=False,
    )
    prepared.frame.to_csv(
        final_path,
        index=False,
    )
    print(f"Auditoría M1: {audit_path}")
    print(f"Muestras finales M1: {final_path}")

    state["prepared_m1"] = prepared
    state["m1_audit_path"] = audit_path
    state["m1_final_path"] = final_path
    return state


def inspect_model_2_candidates(
    state: dict | None = None,
) -> dict:
    state = state or deduplicate_sources()
    if "deduplicated_path" not in state:
        state = deduplicate_sources(state)

    print(
        "\n=== STEP 6 · CANDIDATOS REALES DEL MODELO 2 ==="
    )
    candidates = load_georeferenced_li_samples(
        Path(state["deduplicated_path"])
    )
    _dataset_summary(
        candidates,
        "Muestras con Li + coordenadas válidas para solicitar Sentinel-2",
    )
    if "spatial_group" in candidates.columns:
        print(
            "Grupos espaciales candidatos M2: "
            f"{candidates['spatial_group'].nunique():,}"
        )

    inspection_dir = Path(
        state["paths"]["inspection_dir"]
    )
    inspection_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    destination = (
        inspection_dir
        / "model_2_candidates.csv"
    )
    candidates.to_csv(
        destination,
        index=False,
    )
    print(f"Candidatos M2: {destination}")
    print(
        "Nota: este N es anterior a descargar/verificar "
        "los parches Sentinel-2. El N final de entrenamiento M2 "
        "puede ser menor si faltan escenas válidas."
    )

    state["m2_candidates"] = candidates
    state["m2_candidates_path"] = destination
    return state


def run_all() -> dict:
    state = inspect_sources()
    state = harmonize_georoc(state)
    state = concatenate_sources(state)
    state = deduplicate_sources(state)
    state = inspect_model_1_pipeline(state)
    state = inspect_model_2_candidates(state)

    prepared = state["prepared_m1"]
    candidates = state["m2_candidates"]
    result = state["georoc_result"]
    audit = state["dedupe_audit"]

    summary = {
        "mamani_raw": len(state["base"]),
        "georoc_rows_read": result.rows_read,
        "georoc_compatible": result.rows_kept,
        "concatenated_rows": len(
            state["concatenated"]
        ),
        "duplicates_removed": audit[
            "duplicates_removed"
        ],
        "combined_deduplicated": len(
            state["deduplicated"]
        ),
        "model_1_final": len(
            prepared.frame
        ),
        "model_1_features": (
            len(prepared.schema.numeric)
            + len(prepared.schema.categorical)
        ),
        "model_1_spatial_groups": (
            int(prepared.groups.nunique())
            if prepared.groups is not None
            else 0
        ),
        "model_2_candidates_before_sentinel": len(
            candidates
        ),
    }

    print(
        "\n===================================================="
    )
    print(
        "RESUMEN FINAL · SIN ENTRENAR NINGÚN MODELO"
    )
    print(
        "===================================================="
    )
    _print_table(
        "Conteos finales",
        pd.DataFrame(
            [
                {
                    "métrica": key,
                    "valor": value,
                }
                for key, value in summary.items()
            ]
        ),
    )

    inspection_dir = Path(
        state["paths"]["inspection_dir"]
    )
    summary_path = (
        inspection_dir
        / "pipeline_inspection_summary.json"
    )
    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Resumen JSON: {summary_path}")
    return state


def interactive() -> None:
    while True:
        print(
            "\nINSPECCIÓN DEL PIPELINE DE DATOS"
        )
        print("1. Inspeccionar fuentes brutas")
        print("2. Armonizar GEOROC")
        print("3. Concatenar Mamani09 + GEOROC")
        print("4. Deduplicar dataset combinado")
        print("5. Ejecutar limpieza completa de M1")
        print("6. Obtener candidatos para M2")
        print(
            "7. Ejecutar steps 1-6 y mostrar informe completo"
        )
        print("0. Salir")
        option = choose(
            "\nSeleccione step [0-7]: ",
            {
                "0",
                "1",
                "2",
                "3",
                "4",
                "5",
                "6",
                "7",
            },
        )
        if option == "0":
            return

        actions = {
            "1": inspect_sources,
            "2": harmonize_georoc,
            "3": concatenate_sources,
            "4": deduplicate_sources,
            "5": inspect_model_1_pipeline,
            "6": inspect_model_2_candidates,
            "7": run_all,
        }
        try:
            actions[option]()
        except Exception as exc:
            print(f"\n[ERROR] {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Inspecciona el pipeline de datos de LithiumScope "
            "sin entrenar modelos."
        )
    )
    parser.add_argument(
        "--step",
        choices=[
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "all",
        ],
        help=(
            "Ejecuta un step directamente. "
            "Use 7 o all para ejecutar el flujo completo."
        ),
    )
    args = parser.parse_args()
    if args.step in {"7", "all"}:
        run_all()
        return
    if args.step:
        actions = {
            "1": inspect_sources,
            "2": harmonize_georoc,
            "3": concatenate_sources,
            "4": deduplicate_sources,
            "5": inspect_model_1_pipeline,
            "6": inspect_model_2_candidates,
        }
        actions[args.step]()
        return
    interactive()


if __name__ == "__main__":
    main()
