"""Submenú de simulaciones científicas. No modifica modelos."""
from pathlib import Path

import pandas as pd

from lithiumscope.cli.prompts import choose
from lithiumscope.core.paths import PROJECT_ROOT, RESULTS_DIR
from lithiumscope.persistence.load_model import load_latest_model
from lithiumscope.experiments.scenarios import evaluate_model_1
from lithiumscope.experiments.imagery import evaluate_model_2_image


def run() -> None:
    while True:
        print("\nLABORATORIO DE SIMULACIONES")
        print("1. Modelo 1: escenarios geoquímicos (10 casos demostrativos)")
        print("2. Modelo 2: escenarios espectrales (archivo NPY/NPZ/TIF)")
        print("3. Mostrar informes existentes")
        print("4. Adquirir parches Sentinel-2 reales (piloto)")
        print("5. Ejecutar simulaciones masivas (M1 y M2 disponibles)")
        print("6. Generar comparativa geográfica y temporal (M2)")
        print("0. Volver")
        choice = choose("Seleccione una opción: ", {"0", "1", "2", "3", "4", "5", "6"})
        if choice == "0":
            return
        if choice == "1":
            source = PROJECT_ROOT / "examples" / "integrated_demo" / "cases.csv"
            if not source.is_file():
                print(f"No existe el conjunto de casos: {source}")
                continue
            bundle, path = load_latest_model("model_1")
            print(f"Modelo utilizado: {path}")
            output = evaluate_model_1(
                bundle, pd.read_csv(source),
                RESULTS_DIR / "scientific_simulations" / "model_1",
            )
            print(f"Resultados: {output}")
        elif choice == "2":
            raw = input("Ruta a imagen local (Enter para cancelar): ").strip().strip('"')
            if not raw:
                continue
            from lithiumscope.model_2.steps.step_01_load_imagery import load_imagery
            path = Path(raw)
            if not path.is_file():
                print(f"No existe: {path}")
                continue
            bundle, model_path = load_latest_model("model_2")
            print(f"Modelo utilizado: {model_path}")
            output = evaluate_model_2_image(
                bundle, load_imagery(path),
                RESULTS_DIR / "scientific_simulations" / "model_2",
            )
            print(f"Resultados: {output}")
        elif choice == "4":
            from lithiumscope.experiments.acquisition import acquire_pilot
            destination = RESULTS_DIR / "scientific_simulations" / "sentinel2_real"
            manifest, failures = acquire_pilot(destination)
            print(f"Catálogo: {manifest} | Fallos: {failures}")
        elif choice == "5":
            from lithiumscope.experiments.batch import run_m1_batch, run_m2_batch, summarize
            folder = RESULTS_DIR / "scientific_simulations"
            m1, _ = load_latest_model("model_1")
            source = PROJECT_ROOT / "examples" / "integrated_demo" / "cases.csv"
            result = run_m1_batch(m1, pd.read_csv(source), folder / "batch_m1")
            print(f"M1: {result} | {summarize(result, result.parent)}")
            manifest = folder / "sentinel2_real" / "manifest.csv"
            if manifest.is_file():
                m2, _ = load_latest_model("model_2")
                result = run_m2_batch(m2, manifest, folder / "batch_m2")
                print(f"M2: {result} | {summarize(result, result.parent)}")
            else:
                print("M2 pendiente: primero adquiera parches Sentinel-2 (opción 4).")
        elif choice == "6":
            from lithiumscope.experiments.reporting import compare_geographies
            folder = RESULTS_DIR / "scientific_simulations" / "batch_m2"
            batch = folder / "batch_m2.csv"
            if not batch.exists():
                print("Primero ejecute las simulaciones M2.")
            else:
                csv_file, report_file = compare_geographies(batch, folder)
                print(f"Comparativa: {csv_file} | Resumen: {report_file}")
        else:
            folder = RESULTS_DIR / "scientific_simulations"
            reports = sorted(folder.glob("**/*metrics.csv")) if folder.exists() else []
            if not reports:
                print("Todavía no hay informes generados.")
            for path in reports:
                print(path)
