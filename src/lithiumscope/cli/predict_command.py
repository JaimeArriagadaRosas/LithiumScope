from lithiumscope.cli.file_picker import pick_file
from lithiumscope.cli.prompts import choose
from lithiumscope.core.logger import run_log


def run() -> None:
    from lithiumscope.model_1.prediction.predictor import predict_model_1
    from lithiumscope.model_2.prediction.predictor import predict_model_2

    print("\n1. Modelo 1 — Predecir Li desde datos tabulares")
    print("2. Modelo 2 — Analizar imagen para prospectividad")
    print("0. Volver")
    choice = choose("> ", {"0", "1", "2"})
    if choice == "0":
        return

    if choice == "1":
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
        return

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
