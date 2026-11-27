from __future__ import annotations

from lithiumscope.core.scientific.candidate_evidence import eligible_rows

from pathlib import Path

import pandas as pd

from lithiumscope.core.config import load_config
from lithiumscope.resources.detection.device import DeviceInfo
from lithiumscope.core.logging.session import get_logger
from lithiumscope.core.reproducibility import set_global_seed
from lithiumscope.core.run_context import RunContext
from lithiumscope.core.run_resume import build_training_signature
from lithiumscope.core.states import RunState
from lithiumscope.datasets.provenance.manifest import build_tabular_manifest, write_manifest
from lithiumscope.model_1.evaluation.artifacts import save_dataset_artifacts
from lithiumscope.model_1.evaluation.baseline import evaluate_baselines
from lithiumscope.model_1.preparation.training_data import prepare_training_data
from lithiumscope.model_1.validation.training_preflight import run_model_1_preflight
from lithiumscope.runtime.graceful_shutdown import get_shutdown_manager
from lithiumscope.model_1.training.execution.candidate_evaluation import evaluate_candidates
from lithiumscope.model_1.training.execution.winner_finalization import CompetitionOutcome, finalize_winner

logger = get_logger("model_1.competition")






def _target_experiments(
    algorithms: list[str],
    config: dict,
) -> list[tuple[str, str, str]]:
    configured = config.get(
        "competition",
        {},
    ).get(
        "target_transforms",
        {
            "identity": "all",
        },
    )
    experiments: list[
        tuple[str, str, str]
    ] = []
    for target_transform, selected in (
        configured.items()
    ):
        if selected == "all":
            selected_algorithms = set(
                algorithms
            )
        else:
            selected_algorithms = {
                str(value)
                for value in selected
            }
        for algorithm in algorithms:
            if (
                algorithm
                not in selected_algorithms
            ):
                continue
            variant_id = (
                f"{algorithm}__"
                f"{target_transform}"
            )
            experiments.append(
                (
                    variant_id,
                    algorithm,
                    str(target_transform),
                )
            )
    return experiments






def run_model_1_competition(dataset_path: Path, device: DeviceInfo) -> CompetitionOutcome:
    config = load_config("model_1")
    app_config = load_config("app")
    algorithms = list(config["model"]["algorithms"])
    experiments = _target_experiments(
        algorithms,
        config,
    )
    seed = int(config["validation"]["random_seed"])
    set_global_seed(
        seed,
        deterministic_torch=bool(
            app_config.get("reproducibility", {}).get("deterministic_torch", False)
        ),
    )

    signature = build_training_signature(
        "model_1",
        dataset_path,
        ("app", "model_1"),
    )
    context = RunContext.create(
        "model_1",
        training_signature=signature,
        resume=True,
    )

    print("\n=== MODELO 1 · COMPETENCIA DE ALGORITMOS ===")
    print(f"Entrenamiento: {context.run_id}")
    logger.info("M1_COMPETITION_STARTED run=%s candidates=%d", context.run_id, len(experiments))

    if context.resumed and context.tracker.state == RunState.COMPLETED:
        ranking_path = context.tables / "competition_ranking.csv"
        ranking = pd.read_csv(ranking_path) if ranking_path.exists() else pd.DataFrame()
        summary = context.tracker.payload.get("summary", {})
        winner = summary.get("winner")
        successful = (
            ranking[
                eligible_rows(ranking)
                & (ranking.get("is_baseline") != True)  # noqa: E712
            ]["algorithm"].astype(str).tolist()
            if not ranking.empty
            else []
        )
        print("↻ Entrenamiento compatible ya completado; se reutiliza sin regenerar modelos.")
        print(f"Resultados: {context.root}")
        return CompetitionOutcome(context.root, ranking, winner, successful, [])

    if context.resumed:
        print(f"↻ Reanudando entrenamiento compatible: {context.run_id}")
        context.tracker.event(
            "run_resumed",
            previous_state=context.tracker.state.value,
        )
    context.tracker.set_state(RunState.RUNNING)
    get_shutdown_manager().register_cleanup(context.tracker.cancel_if_active)
    print("Preparando dataset común...")

    prepared = prepare_training_data(dataset_path, model_family="common")
    save_dataset_artifacts(prepared, context)
    preflight = run_model_1_preflight(
        prepared,
        config,
        context,
    )

    from lithiumscope.model_1.eda.pipeline import record_training_eda
    record_training_eda(dataset_path, prepared, config, context)
    dataset_manifest = build_tabular_manifest(
        dataset_path,
        dataset_name="model_1_training",
        model_group="model_1",
        stage="processed_training_input",
        frame=prepared.frame,
        metadata={
            "raw_rows": len(prepared.raw_frame),
            "processed_rows": len(prepared.frame),
            "target": prepared.target,
        },
    )
    manifest_path = write_manifest(
        dataset_manifest,
        context.manifests / "dataset_manifest.json",
    )
    context.tracker.attach_summary(dataset_sha256=dataset_manifest.source_sha256)

    baseline = evaluate_baselines(prepared, config)
    baseline[1].to_csv(context.tables / "baseline_fold_metrics.csv", index=False)
    baseline[2].to_csv(context.tables / "baseline_oof_predictions.csv", index=False)
    rows: list[dict] = list(baseline[0])
    results: dict[str, tuple] = {}
    failed: list[str] = []

    evaluate_candidates(config, context, device, experiments, failed, prepared, results, rows)

    return finalize_winner(baseline, config, context, dataset_manifest, device, failed, manifest_path, preflight, prepared, results, rows, seed)
