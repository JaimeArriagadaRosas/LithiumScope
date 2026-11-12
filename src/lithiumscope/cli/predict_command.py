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
    print(f"  Excel:     {result.workbook_path}")
    print(f"  Manifest:  {result.manifest_path}")
    if result.session.mode == "demonstration":
        print(f"  Imágenes:  {result.session.model_2 / 'case_imagery_status.csv'}")
    if result.report_path is None:
        print("  PDF:       no generado; las predicciones quedaron guardadas")
        print("  Para renderizar más adelante, desde la raíz del proyecto:")
        print('    set "PYTHONPATH=%CD%\\src"')
        print(
            "    python -m lithiumscope.prediction.rebuild_pdf "
            f'--run "{result.session.run_id}"'
        )
    else:
        print(f"  Reporte:   {result.report_path}")
        if open_report_in_default_browser(result.report_path):
            print("  Apertura: PDF enviado al navegador predeterminado")
        else:
            print("  Apertura: PDF guardado; no fue posible abrir el navegador")

    if result.session.mode == "demonstration":
        for label, frame, columns in (
            (
                "MODELO 1 — CONCENTRACIÓN",
                result.model_1_predictions,
                ["case_id", "Li_icpms", "Li_icpms_predicted", "applicability_warning"],
            ),
            (
                "MODELO 2 — PROSPECTIVIDAD",
                result.model_2_predictions,
                ["case_id", "prospectivity_score", "priority",
                 "quality_status", "applicability_warning"],
            ),
        ):
            print(f"\n{label}")
            selected = [column for column in columns if column in frame.columns]
            print(frame[selected].to_string(index=False) if selected else "Sin casos evaluables")
        print("\nResultados experimentales: no se activó ningún candidato.")


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
    from lithiumscope.core.config import load_config
    from lithiumscope.prediction.integrated import run_automatic_demonstration

    versions = load_config("prediction")["demonstration"]["candidate_models"]
    print("\nDEMOSTRACIÓN INTEGRADA — 10 CASOS")
    print(f"1. Candidatos nuevos: M1 {versions['model_1']} / M2 {versions['model_2']}")
    print("   Solo evaluación; NO activa ni publica modelos.")
    print("2. Modelos activos del release anterior")
    print("0. Volver")
    mode = choose("Seleccione modelos [0-2]: ", {"0", "1", "2"})
    if mode == "0":
        return

    print("\n1. Generar Excel, CSV, evidencias y PDF")
    print("2. Generar Excel, CSV y evidencias SIN PDF (renderizable después)")
    render_choice = choose("Seleccione salida [1-2]: ", {"1", "2"})

    result = run_automatic_demonstration(
        model_1_candidate_run_id=versions["model_1"] if mode == "1" else None,
        model_2_candidate_run_id=versions["model_2"] if mode == "1" else None,
        render_pdf=render_choice == "1",
    )
    _print_integrated_result(result)


def run() -> None:
    print("\nREALIZAR PREDICCIÓN")
    print("1. Modelo 1 — Predicción de concentración")
    print("2. Modelo 2 — Prospectividad espacial")
    print("3. Predicción completa")
    print("4. Demostración integrada automática")
    print("0. Volver")
    choice = choose("> ", {"0", "1", "2", "3", "4"})

    if choice == "0":
        return False

    if choice == "1":
        _run_model_1()
    elif choice == "2":
        _run_model_2()
    elif choice == "3":
        _run_complete()
    elif choice == "4":
        _run_demonstration()
