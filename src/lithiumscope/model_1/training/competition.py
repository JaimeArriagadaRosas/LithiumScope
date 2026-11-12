from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from lithiumscope.core.config import load_config
from lithiumscope.core.device import DeviceInfo
from lithiumscope.core.logger import get_logger
from lithiumscope.core.reproducibility import set_global_seed
from lithiumscope.core.run_context import RunContext
from lithiumscope.core.training_diagnostics import save_oof_diagnostics
from lithiumscope.core.run_resume import build_training_signature
from lithiumscope.core.states import RunState
from lithiumscope.datasets.manifest import build_tabular_manifest, write_manifest
from lithiumscope.model_1.evaluation.artifacts import save_algorithm_artifacts, save_dataset_artifacts
from lithiumscope.model_1.evaluation.baseline import evaluate_mean_baseline
from lithiumscope.model_1.evaluation.importance import save_feature_importance
from lithiumscope.model_1.evaluation.plots import save_competition_chart
from lithiumscope.model_1.pipeline import prepare_training_data
from lithiumscope.model_1.preflight import run_model_1_preflight
from lithiumscope.model_1.schema import applicability_profile
from lithiumscope.model_1.training.cv_runner import run_nested_cv
from lithiumscope.model_1.training.factory import build_pipeline, get_algorithm
from lithiumscope.persistence.save_model import save_model_bundle
from lithiumscope.results.dashboard import build_dashboard
from lithiumscope.results.excel_exporter import export_workbook
from lithiumscope.runtime.graceful_shutdown import get_shutdown_manager

logger = get_logger("model_1.competition")


@dataclass
class CompetitionOutcome:
    run_dir: Path
    ranking: pd.DataFrame
    winner: str | None
    successful: list[str]
    failed: list[str]


def _ranking_row(result) -> dict:
    folds = result.fold_table
    return {
        "algorithm": (
            result.variant_id
            or result.algorithm
        ),
        "base_algorithm": result.algorithm,
        "target_transform": result.target_transform,
        "label": result.label,
        "rmse_mean": float(folds["rmse"].mean()),
        "rmse_std": float(folds["rmse"].std(ddof=0)),
        "mae_mean": float(folds["mae"].mean()),
        "mae_std": float(folds["mae"].std(ddof=0)),
        "r2_mean": float(folds["r2"].mean()),
        "r2_std": float(folds["r2"].std(ddof=0)),
        "median_ae_mean": float(folds["median_ae"].mean()),
        "explained_variance_mean": float(folds["explained_variance"].mean()),
        "elapsed_seconds": result.elapsed_seconds,
        "status": "ok",
        "is_baseline": False,
    }


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


def _rank(rows: list[dict]) -> pd.DataFrame:
    ranking = pd.DataFrame(rows)
    if ranking.empty:
        return ranking

    baseline = ranking[ranking.get("is_baseline", False) == True].copy()  # noqa: E712
    candidates = ranking[
        (ranking["status"] == "ok")
        & (ranking.get("is_baseline", False) != True)  # noqa: E712
    ].copy()
    failed = ranking[ranking["status"] != "ok"].copy()

    if not candidates.empty:
        candidates = candidates.sort_values(
            ["rmse_mean", "mae_mean", "r2_mean"],
            ascending=[True, True, False],
        ).reset_index(drop=True)
        candidates.insert(0, "rank", range(1, len(candidates) + 1))

    if not baseline.empty:
        baseline.insert(0, "rank", pd.NA)

    return pd.concat([candidates, baseline, failed], ignore_index=True, sort=False)


def _excel_sheets(prepared, ranking: pd.DataFrame, results: dict[str, tuple], baseline) -> dict[str, pd.DataFrame]:
    sheets = {
        "ranking": ranking,
        "pipeline": prepared.audit.to_frame(),
        "dataset_summary": prepared.frame.describe(include="all").transpose().reset_index(names="variable"),
        "correlations": prepared.frame.select_dtypes(include=[np.number]).corr(numeric_only=True).reset_index(),
        "baseline_folds": baseline[1],
    }
    for algorithm, (result, algorithm_prepared) in results.items():
        sheets[f"folds_{algorithm}"[:31]] = result.fold_table
        sheets[f"pred_{algorithm}"[:31]] = pd.DataFrame(
            {
                "row_index": algorithm_prepared.y.index,
                "Li_real": algorithm_prepared.y.to_numpy(),
                "Li_pred": result.predictions,
                "residual": algorithm_prepared.y.to_numpy() - result.predictions,
            }
        )
    return sheets


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

    if context.resumed and context.tracker.state == RunState.COMPLETED:
        ranking_path = context.tables / "competition_ranking.csv"
        ranking = pd.read_csv(ranking_path) if ranking_path.exists() else pd.DataFrame()
        summary = context.tracker.payload.get("summary", {})
        winner = summary.get("winner")
        successful = (
            ranking[
                (ranking.get("status") == "ok")
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

    baseline = evaluate_mean_baseline(prepared, config)
    baseline[1].to_csv(context.tables / "baseline_fold_metrics.csv", index=False)
    rows: list[dict] = [baseline[0]]
    results: dict[str, tuple] = {}
    failed: list[str] = []

    _evaluate_model_1_candidates(config, context, device, experiments, failed, prepared, results, rows)

    return _finalize_model_1_winner(baseline, config, context, dataset_manifest, device, failed, manifest_path, preflight, prepared, results, rows, seed)

from lithiumscope.model_1.training.competition_evaluate_model_1_candidates import _evaluate_model_1_candidates
from lithiumscope.model_1.training.competition_finalize_model_1_winner import _finalize_model_1_winner
