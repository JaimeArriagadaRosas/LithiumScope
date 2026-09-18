from __future__ import annotations

from pathlib import Path

from lithiumscope.cli.display import device_summary
from lithiumscope.cli.file_picker import pick_file
from lithiumscope.cli.prompts import choose
from lithiumscope.core.device import detect_device
from lithiumscope.core.logger import get_logger, run_log
from lithiumscope.datasets.downloader import ensure_dataset
from lithiumscope.runtime.preboot import require_full_training_environment

logger = get_logger("cli.train")


def _model_1_dataset() -> Path | None:
    try:
        return ensure_dataset("mamani09_public_mirror")
    except Exception as exc:
        logger.exception("Automatic Model 1 dataset acquisition failed")
        print(f"\nDescarga automática falló: {exc}")
        return pick_file(
            "Seleccione dataset para Modelo 1",
            [("Datos tabulares", "*.csv *.xlsx *.xls"), ("Todos", "*.*")],
        )


def _model_2_manifest() -> Path | None:
    default = Path("data/processed/model_2/training_manifest.csv")
    if default.exists():
        return default
    print(
        "\nModelo 2 requiere un manifiesto muestra-imagen con image_path, Li_icpms "
        "y, de forma recomendada, spatial_group."
    )
    return pick_file(
        "Seleccione training_manifest.csv para Modelo 2",
        [("CSV", "*.csv"), ("Todos", "*.*")],
    )


def _print_outcome(outcome, model_name: str) -> None:
    print(f"\n{model_name} completado.")
    print(f"Resultados: {outcome.run_dir}")
    if outcome.winner:
        print(f"Ganador provisional: {outcome.winner}")
    if outcome.failed:
        print("Algoritmos omitidos/fallidos: " + ", ".join(outcome.failed))
    if outcome.ranking.empty:
        return
    allowed = {
        "rank", "label", "rmse_mean", "mae_mean", "r2_mean",
        "roc_auc_mean", "average_precision_mean", "balanced_accuracy_mean", "status",
    }
    columns = [column for column in outcome.ranking.columns if column in allowed]
    print("\nRanking:")
    print(outcome.ranking[columns].to_string(index=False))


def run() -> None:
    from lithiumscope.model_1.training.competition import run_model_1_competition
    from lithiumscope.model_2.training.competition import run_model_2_competition

    print("\n1. Modelo 1 — Competencia de predicción de Li")
    print("2. Modelo 2 — Competencia de prospectividad espacial")
    print("3. Entrenar ambos")
    print("0. Volver")
    choice = choose("> ", {"0", "1", "2", "3"})
    if choice == "0":
        return

    # A full competition is all-or-nothing: detect every missing ML dependency
    # before spending time on the first algorithm.
    require_full_training_environment()

    device = detect_device(prefer_gpu=True)
    device_summary(device)

    if choice in {"1", "3"}:
        dataset = _model_1_dataset()
        if dataset is None:
            print("Modelo 1 cancelado.")
        else:
            print("\nOrden: RF → XGBoost → SVM → TabNet → HistGradientBoosting → CatBoost")
            with run_log("training", "model_1_competition"):
                outcome = run_model_1_competition(dataset, device)
            _print_outcome(outcome, "Modelo 1")

    if choice in {"2", "3"}:
        manifest = _model_2_manifest()
        if manifest is None:
            print("Modelo 2 cancelado.")
        else:
            print("\nOrden: RF → Extra Trees → HistGradientBoosting → XGBoost → CatBoost → SVM-RBF")
            with run_log("training", "model_2_competition"):
                outcome = run_model_2_competition(manifest, device)
            _print_outcome(outcome, "Modelo 2")
