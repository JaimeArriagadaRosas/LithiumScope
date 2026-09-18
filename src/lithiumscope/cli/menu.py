from __future__ import annotations

from pathlib import Path

from lithiumscope.cli.display import device_summary, header, pause
from lithiumscope.cli.file_picker import pick_file
from lithiumscope.core.device import detect_device
from lithiumscope.core.logger import configure_logging, get_logger, run_log
from lithiumscope.core.paths import ensure_runtime_directories
from lithiumscope.datasets.downloader import ensure_dataset
from lithiumscope.persistence.model_registry import list_metadata

logger = get_logger("cli")


def _choose(prompt: str, allowed: set[str]) -> str:
    while True:
        try:
            value = input(prompt).strip()
        except EOFError:
            return "0"
        if value in allowed:
            return value
        print("Opción no válida.")


def _print_training_result(report: dict) -> None:
    print("\nEntrenamiento completado.")
    print(f"Modelo: {report.get('model_group')} / {report.get('algorithm')}")
    print(f"Muestras: {report.get('samples')}")
    metrics = report.get("metrics", {})
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            print(f"{key.upper()}: {value:.4f}")
    print(f"Modelo guardado: {report.get('model_path')}")


def train_menu() -> None:
    from lithiumscope.model_1.training.trainer import train_model_1
    from lithiumscope.model_2.training.trainer import train_model_2

    print("\n1. Modelo 1 — Predicción de Li")
    print("2. Modelo 2 — Prospectividad espacial")
    print("3. Entrenar ambos")
    print("0. Volver")
    choice = _choose("> ", {"0", "1", "2", "3"})
    if choice == "0":
        return

    device = detect_device(prefer_gpu=True)
    device_summary(device)

    if choice in {"1", "3"}:
        print("\nAlgoritmo Modelo 1:")
        print("1. Random Forest")
        print("2. XGBoost")
        print("3. SVM")
        print("4. TabNet")
        algorithm_choice = _choose("> ", {"1", "2", "3", "4"})
        algorithm = {
            "1": "random_forest",
            "2": "xgboost",
            "3": "svm",
            "4": "tabnet",
        }[algorithm_choice]

        try:
            dataset_path = ensure_dataset("mamani09_public_mirror")
        except Exception as exc:
            logger.exception("Automatic Model 1 dataset acquisition failed")
            print(f"\nDescarga automática falló: {exc}")
            dataset_path = pick_file(
                "Seleccione dataset para Modelo 1",
                [("Datos tabulares", "*.csv *.xlsx *.xls"), ("Todos", "*.*")],
            )
            if dataset_path is None:
                print("Entrenamiento cancelado.")
                return

        with run_log("training", f"model_1_{algorithm}"):
            report = train_model_1(dataset_path, algorithm=algorithm, device=device)
        _print_training_result(report)

    if choice in {"2", "3"}:
        manifest = Path("data/processed/model_2/training_manifest.csv")
        if not manifest.exists():
            print(
                "\nModelo 2 necesita un manifiesto que relacione cada muestra conocida "
                "con su image_path y Li_icpms."
            )
            selected = pick_file(
                "Seleccione training_manifest.csv para Modelo 2",
                [("CSV", "*.csv"), ("Todos", "*.*")],
            )
            if selected is None:
                print("Modelo 2 omitido: aún no hay pares muestra-imagen preparados.")
                return
            manifest = selected

        with run_log("training", "model_2_prospectivity"):
            report = train_model_2(manifest, device=device)
        _print_training_result(report)


def prediction_menu() -> None:
    from lithiumscope.model_1.prediction.predictor import predict_model_1
    from lithiumscope.model_2.prediction.predictor import predict_model_2

    print("\n1. Modelo 1 — Predecir Li desde datos tabulares")
    print("2. Modelo 2 — Analizar imagen para prospectividad")
    print("0. Volver")
    choice = _choose("> ", {"0", "1", "2"})
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

    if choice == "2":
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


def metrics_menu() -> None:
    print("\n1. Modelo 1")
    print("2. Modelo 2")
    print("3. Ambos")
    print("0. Volver")
    choice = _choose("> ", {"0", "1", "2", "3"})
    if choice == "0":
        return
    groups = []
    if choice in {"1", "3"}:
        groups.append("model_1")
    if choice in {"2", "3"}:
        groups.append("model_2")

    for group in groups:
        print(f"\n{group.upper()}")
        records = list_metadata(group)
        if not records:
            print("Sin entrenamientos registrados.")
            continue
        for record in records[:5]:
            print("-" * 48)
            print(f"Algoritmo: {record.get('algorithm')}")
            print(f"Guardado: {record.get('saved_at_utc')}")
            print(f"Muestras: {record.get('samples')}")
            metrics = record.get("metrics", {})
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    print(f"{key.upper()}: {value:.4f}")


def main() -> int:
    ensure_runtime_directories()
    configure_logging()
    logger.info("LithiumScope started")

    while True:
        header()
        print("1. Entrenar modelos")
        print("2. Realizar predicción")
        print("3. Métricas y resultados")
        print("0. Salir")
        choice = _choose("\nSeleccione una opción: ", {"0", "1", "2", "3"})

        try:
            if choice == "1":
                train_menu()
                pause()
            elif choice == "2":
                prediction_menu()
                pause()
            elif choice == "3":
                metrics_menu()
                pause()
            else:
                logger.info("LithiumScope finished")
                return 0
        except KeyboardInterrupt:
            print("\nOperación cancelada.")
            logger.warning("Operation cancelled by user")
            pause()
        except Exception as exc:
            logger.exception("Unhandled application error")
            print(f"\nError: {exc}")
            print("Revise logs/errors/errors.log para el detalle.")
            pause()
