from pathlib import Path

from lithiumscope.cli.file_picker import pick_file
from lithiumscope.cli.prompts import choose
from lithiumscope.cli.report_viewer import open_report_in_default_browser
from lithiumscope.core.logger import run_log
from lithiumscope.prediction.interpretation import (
    model_1_case_text,
    model_2_case_text,
)


def _print_integrated_result(result) -> None:
    print("\n" + result.interpretation)
    print("\nARTEFACTOS")
    print(f"  Ejecución: {result.session.root}")
    print(f"  Reporte:   {result.report_path}")
    print(f"  Excel:     {result.workbook_path}")
    print(f"  Manifest:  {result.manifest_path}")
    if open_report_in_default_browser(result.report_path):
        print("  Apertura:   reporte PDF enviado al navegador predeterminado")
    else:
        print("  Apertura:   no fue posible abrir el navegador automáticamente")


def _run_model_1() -> None:
    from lithiumscope.model_1.prediction.predictor import predict_model_1

    path = pick_file(
        "Seleccione Excel/CSV para predecir Li",
        [("Datos tabulares", "*.xlsx *.xls *.csv"), ("Todos", "*.*")],
    )
    if path is None:
        return
    with run_log("prediction", "model_1"):
        output, destination, model_path = predict_model_1(path)
    print(f"\nModelo: {model_path}")
    print(f"Predicciones: {destination}")
    print(output[["Li_icpms_predicted"]].head(10).to_string(index=False))
    if not output.empty:
        print("\nInterpretación:")
        print(model_1_case_text(output.iloc[0]))


def _run_model_2() -> None:
    from lithiumscope.model_2.prediction.predictor import predict_model_2

    path = pick_file(
        "Seleccione imagen multibanda",
        [("GeoTIFF / NumPy", "*.tif *.tiff *.npy"), ("Todos", "*.*")],
    )
    if path is None:
        return
    with run_log("prediction", "model_2"):
        payload, destination, model_path = predict_model_2(path)
    print(f"\nModelo: {model_path}")
    print(f"Prospectividad: {payload['prospectivity_score']:.3f}")
    print(f"Prioridad: {payload['priority'].upper()}")
    print(f"Resultado: {destination}")
    print("\nInterpretación:")
    import pandas as pd
    print(model_2_case_text(pd.Series(payload)))


def _run_complete() -> None:
    from lithiumscope.prediction.integrated import run_complete_prediction

    model_1_path = pick_file(
        "Predicción completa · seleccione datos tabulares para Modelo 1",
        [("Datos tabulares", "*.xlsx *.xls *.csv"), ("Todos", "*.*")],
    )
    if model_1_path is None:
        return
    model_2_path = pick_file(
        "Predicción completa · seleccione imagen o manifest para Modelo 2",
        [
            ("Imagen o manifest", "*.tif *.tiff *.npy *.csv *.xlsx *.xls"),
            ("Todos", "*.*"),
        ],
    )
    if model_2_path is None:
        return
    result = run_complete_prediction(
        Path(model_1_path),
        Path(model_2_path),
    )
    _print_integrated_result(result)


def _run_demonstration() -> None:
    from lithiumscope.prediction.integrated import run_automatic_demonstration

    print(
        "\nLithiumScope ejecutará automáticamente el conjunto externo versionado, "
        "preparará Sentinel-2 y evaluará ambos modelos."
    )
    result = run_automatic_demonstration()
    _print_integrated_result(result)


def run() -> None:
    print("\nREALIZAR PREDICCIÓN")
    print("1. Modelo 1 — Predicción de concentración")
    print("2. Modelo 2 — Prospectividad espacial")
    print("3. Predicción completa")
    print("4. Demostración integrada automática")
    print("0. Volver")
    choice = choose("> ", {"0", "1", "2", "3", "4"})

    if choice == "1":
        _run_model_1()
    elif choice == "2":
        _run_model_2()
    elif choice == "3":
        _run_complete()
    elif choice == "4":
        _run_demonstration()
