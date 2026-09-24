from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from lithiumscope.cli.prompts import choose
from lithiumscope.tools.lab_preboot import (
    print_gpu_probe,
    run_lab_preboot,
)
from lithiumscope.tools.pipeline_inspection_models import (
    inspect_model_1_pipeline,
    inspect_model_2_candidates,
)
from lithiumscope.tools.pipeline_inspection_reporting import (
    print_table,
)
from lithiumscope.tools.pipeline_inspection_sources import (
    concatenate_sources,
    deduplicate_sources,
    harmonize_georoc,
    inspect_sources,
)


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
        "concatenated_rows": len(state["concatenated"]),
        "duplicates_removed": audit["duplicates_removed"],
        "combined_deduplicated": len(state["deduplicated"]),
        "model_1_final": len(prepared.frame),
        "model_1_features": (
            len(prepared.schema.numeric)
            + len(prepared.schema.categorical)
        ),
        "model_1_spatial_groups": (
            int(prepared.groups.nunique())
            if prepared.groups is not None
            else 0
        ),
        "model_2_candidates_before_sentinel": len(candidates),
    }

    print("\n====================================================")
    print("RESUMEN FINAL · SIN ENTRENAR NINGÚN MODELO")
    print("====================================================")
    print_table(
        "Conteos finales",
        pd.DataFrame(
            [
                {"métrica": key, "valor": value}
                for key, value in summary.items()
            ]
        ),
    )

    inspection_dir = Path(state["paths"]["inspection_dir"])
    summary_path = inspection_dir / "pipeline_inspection_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Resumen JSON: {summary_path}")
    return state


def _actions() -> dict[str, object]:
    return {
        "1": inspect_sources,
        "2": harmonize_georoc,
        "3": concatenate_sources,
        "4": deduplicate_sources,
        "5": inspect_model_1_pipeline,
        "6": inspect_model_2_candidates,
        "7": run_all,
    }


def interactive() -> None:
    while True:
        print("\nINSPECCIÓN DEL PIPELINE DE DATOS")
        print("1. Inspeccionar fuentes brutas")
        print("2. Armonizar GEOROC")
        print("3. Concatenar Mamani09 + GEOROC")
        print("4. Deduplicar dataset combinado")
        print("5. Ejecutar limpieza completa de M1")
        print("6. Obtener candidatos para M2")
        print("7. Ejecutar steps 1-6 y mostrar informe completo")
        print("0. Salir")
        option = choose(
            "\nSeleccione step [0-7]: ",
            {"0", "1", "2", "3", "4", "5", "6", "7"},
        )
        if option == "0":
            return
        try:
            _actions()[option]()
        except Exception as exc:
            print(f"\n[ERROR] {exc}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Inspecciona el pipeline de datos de LithiumScope "
            "sin entrenar modelos."
        )
    )
    parser.add_argument(
        "--step",
        choices=["1", "2", "3", "4", "5", "6", "7", "all"],
        help=(
            "Ejecuta un step directamente. "
            "Use 7 o all para ejecutar el flujo completo."
        ),
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help=(
            "Ejecuta únicamente el diagnóstico GPU. "
            "No descarga datasets."
        ),
    )
    return parser


def main() -> None:
    args = _parser().parse_args()

    if args.gpu:
        raise SystemExit(print_gpu_probe())

    try:
        run_lab_preboot(
            acquire_data=True,
            verbose=True,
        )
    except Exception as exc:
        print(f"\n[ERROR PREBOOT LAB] {exc}")
        return

    if args.step in {"7", "all"}:
        run_all()
        return
    if args.step:
        _actions()[args.step]()
        return
    interactive()


if __name__ == "__main__":
    main()
