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
        print("0. Volver")
        choice = choose("Seleccione una opción: ", {"0", "1", "2", "3"})
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
        else:
            folder = RESULTS_DIR / "scientific_simulations"
            reports = sorted(folder.glob("**/*metrics.csv")) if folder.exists() else []
            if not reports:
                print("Todavía no hay informes generados.")
            for path in reports:
                print(path)
