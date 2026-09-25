from __future__ import annotations

from lithiumscope.cli.display import device_summary
from lithiumscope.cli.prompts import choose
from lithiumscope.core.device import detect_device
from lithiumscope.core.logger import run_log
from lithiumscope.datasets.provisioner import (
    require_model_1_dataset,
    require_model_2_dataset,
)
from lithiumscope.runtime.preboot import (
    require_full_training_environment,
)


def _print_outcome(outcome, model_name: str) -> None:
    print(f"\n{model_name} completado.")
    print(f"Resultados: {outcome.run_dir}")
    if outcome.winner:
        print(
            f"Ganador candidato de esta ejecución: {outcome.winner}"
        )
        print(
            "El modelo activo no se reemplaza durante el entrenamiento. "
            "Evalúe el candidato desde Métricas y resultados."
        )
    if outcome.failed:
        print(
            "Algoritmos omitidos/fallidos: "
            + ", ".join(outcome.failed)
        )
    if outcome.ranking.empty:
        return

    allowed = {
        "rank",
        "label",
        "rmse_mean",
        "mae_mean",
        "r2_mean",
        "roc_auc_mean",
        "average_precision_mean",
        "balanced_accuracy_mean",
        "status",
    }
    columns = [
        column
        for column in outcome.ranking.columns
        if column in allowed
    ]
    print("\nRanking:")
    print(
        outcome.ranking[
            columns
        ].to_string(index=False)
    )


def run() -> None:
    from lithiumscope.model_1.training.competition import (
        run_model_1_competition,
    )
    from lithiumscope.model_2.training.competition import (
        run_model_2_competition,
    )

    print("\nENTRENAMIENTO")
    print("1. Modelo 1 — Competencia de predicción de Li")
    print(
        "2. Modelo 2 — "
        "Competencia de prospectividad espacial"
    )
    print("3. Entrenar ambos")
    print("0. Volver")
    choice = choose(
        "\nSeleccione entrenamiento [0-3]: ",
        {"0", "1", "2", "3"},
    )
    if choice == "0":
        return

    require_full_training_environment()
    device = detect_device(prefer_gpu=True)
    device_summary(device)

    print(
        "\nFuente de entrenamiento: Mamani09 + GEOROC. "
        "GEOROC se armoniza y deduplica antes de entrenar."
    )

    if choice in {"1", "3"}:
        model_1_dataset = require_model_1_dataset(
            include_georoc=True
        )
        print(
            "\nOrden Modelo 1: "
            "RF → XGBoost → SVM → TabNet → "
            "HistGradientBoosting → CatBoost"
        )
        with run_log(
            "training",
            "model_1_competition",
        ) as training_log:
            outcome = run_model_1_competition(
                model_1_dataset,
                device,
            )
        print(
            f"Log detallado Modelo 1: {training_log}"
        )
        _print_outcome(
            outcome,
            "Modelo 1",
        )

    if choice in {"2", "3"}:
        model_2_manifest = require_model_2_dataset(
            include_georoc=True
        )
        print(
            "\nOrden Modelo 2: "
            "RF → Extra Trees → HistGradientBoosting → "
            "XGBoost → CatBoost → SVM-RBF"
        )
        with run_log(
            "training",
            "model_2_competition",
        ):
            outcome = run_model_2_competition(
                model_2_manifest,
                device,
            )
        _print_outcome(
            outcome,
            "Modelo 2",
        )
