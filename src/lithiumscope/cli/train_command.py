from __future__ import annotations

from lithiumscope.cli.display import device_summary
from lithiumscope.cli.prompts import choose
from lithiumscope.core.device import detect_device
from lithiumscope.core.logger import run_log
from lithiumscope.core.training_logs import training_session, register_training_run
from lithiumscope.datasets.provisioner import (
    require_model_1_dataset,
    require_model_2_expanded_manifest,
)
from lithiumscope.runtime.preboot import (
    require_full_training_environment,
)


def _print_outcome(outcome, model_name: str) -> None:
    print(f"\n{model_name} completado.")
    print(f"Resultados: {outcome.run_dir}")
    if outcome.winner:
        print(
            f"Ganador candidato de esta ejecuci├│n: {outcome.winner}"
        )
        print(
            "El modelo activo no se reemplaza durante el entrenamiento. "
            "Eval├║e el candidato desde M├®tricas y resultados."
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


def _run_impl() -> None:
    from lithiumscope.model_1.training.competition import (
        run_model_1_competition,
    )
    from lithiumscope.model_2.training.competition import (
        run_model_2_competition,
    )

    print("\nENTRENAMIENTO")
    print("1. Modelo 1 ÔÇö Competencia de predicci├│n de Li")
    print(
        "2. Modelo 2 ÔÇö "
        "Competencia de prospectividad espacial"
    )
    print("3. Entrenar ambos")
    print("0. Volver")
    choice = choose(
        "\nSeleccione entrenamiento [0-3]: ",
        {"0", "1", "2", "3"},
    )
    if choice == "0":
        return False

    require_full_training_environment()
    device = detect_device(prefer_gpu=True)
    device_summary(device)

    if choice in {"1", "3"}:
        print("\nModelo 1: candidato B conservado; sin cambios de datos.")
    if choice in {"2", "3"}:
        print(
            "\nModelo 2: ampliacion GEOROC/Sentinel-2 filtrada y reanudable; "
            "706 muestras anteriores preservadas."
        )

    if choice in {"1", "3"}:
        model_1_dataset = require_model_1_dataset(
            include_georoc=True
        )
        print(
            "\nOrden Modelo 1: "
            "RF ÔåÆ XGBoost ÔåÆ SVM ÔåÆ TabNet ÔåÆ "
            "HistGradientBoosting ÔåÆ CatBoost"
        )
        with run_log(
            "training",
            "model_1_competition",
        ) as training_log:
            outcome = run_model_1_competition(
                model_1_dataset,
                device,
            )
        register_training_run("model_1", outcome.run_dir)
        print(
            f"Log detallado Modelo 1: {training_log}"
        )
        _print_outcome(
            outcome,
            "Modelo 1",
        )

    if choice in {"2", "3"}:
        model_2_manifest = require_model_2_expanded_manifest()
        print(
            "\nModelo 2: RF, Extra Trees, HGB, "
            "XGBoost, CatBoost y SVM-RBF."
        )
        with run_log(
            "training",
            "model_2_competition",
        ):
            outcome = run_model_2_competition(
                model_2_manifest,
                device,
            )
        register_training_run("model_2", outcome.run_dir)
        _print_outcome(
            outcome,
            "Modelo 2",
        )



def run() -> None:
    # Capture preparation, both model runs, and failures in a single session.
    with training_session("interactive"):
        return _run_impl()
