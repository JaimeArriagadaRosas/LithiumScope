from __future__ import annotations

from pathlib import Path

import pandas as pd

from lithiumscope.model_1.pipeline import prepare_training_data
from lithiumscope.model_2.data.sample_source import (
    load_georeferenced_li_samples,
)
from lithiumscope.tools.pipeline_inspection_reporting import (
    dataset_summary,
    print_table,
)
from lithiumscope.tools.pipeline_inspection_sources import (
    deduplicate_sources,
)


def inspect_model_1_pipeline(
    state: dict | None = None,
) -> dict:
    state = state or deduplicate_sources()
    if "deduplicated_path" not in state:
        state = deduplicate_sources(state)

    print("\n=== STEP 5 · LIMPIEZA REAL DEL MODELO 1 ===")
    prepared = prepare_training_data(
        Path(state["deduplicated_path"]),
        model_family="random_forest",
    )

    audit = prepared.audit.to_frame()
    print_table(
        "Filas después de cada step M1",
        audit,
    )
    dataset_summary(
        prepared.frame,
        "Dataset final que entra a M1",
    )

    schema_rows = pd.DataFrame(
        [
            {
                "type": "numeric",
                "count": len(prepared.schema.numeric),
                "columns": ", ".join(prepared.schema.numeric),
            },
            {
                "type": "categorical",
                "count": len(prepared.schema.categorical),
                "columns": ", ".join(prepared.schema.categorical),
            },
        ]
    )
    print_table(
        "Esquema final de features M1",
        schema_rows,
    )

    inspection_dir = Path(
        state["paths"]["inspection_dir"]
    )
    inspection_dir.mkdir(parents=True, exist_ok=True)
    audit_path = inspection_dir / "model_1_pipeline_audit.csv"
    final_path = inspection_dir / "model_1_final_samples.csv"
    audit.to_csv(audit_path, index=False)
    prepared.frame.to_csv(final_path, index=False)

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

    print("\n=== STEP 6 · CANDIDATOS REALES DEL MODELO 2 ===")
    candidates = load_georeferenced_li_samples(
        Path(state["deduplicated_path"])
    )
    dataset_summary(
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
    inspection_dir.mkdir(parents=True, exist_ok=True)
    destination = inspection_dir / "model_2_candidates.csv"
    candidates.to_csv(destination, index=False)
    print(f"Candidatos M2: {destination}")
    print(
        "Nota: este N es anterior a descargar/verificar "
        "los parches Sentinel-2. El N final de entrenamiento M2 "
        "puede ser menor si faltan escenas válidas."
    )

    state["m2_candidates"] = candidates
    state["m2_candidates_path"] = destination
    return state
