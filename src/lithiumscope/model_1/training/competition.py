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

    for index, (
        variant_id,
        algorithm,
        target_transform,
    ) in enumerate(
        experiments,
        start=1,
    ):
        base_label = get_algorithm(
            algorithm
        ).label
        display_label = (
            base_label
            if target_transform == "identity"
            else (
                f"{base_label} · "
                f"{target_transform}"
            )
        )
        print(
            f"\n[{index}/{len(experiments)}] "
            f"Entrenando {display_label}"
        )
        context.tracker.event(
            "algorithm_started",
            algorithm=algorithm,
            variant_id=variant_id,
            target_transform=target_transform,
        )

        try:
            algorithm_prepared = prepared

            result = run_nested_cv(
                algorithm_prepared,
                algorithm,
                device,
                config,
                checkpoint_root=context.checkpoints,
                target_transform=target_transform,
                variant_id=variant_id,
            )
            results[variant_id] = (
                result,
                algorithm_prepared,
            )
            save_algorithm_artifacts(
                algorithm_prepared,
                result,
                context,
            )
            rows.append(_ranking_row(result))
            context.tracker.event(
                "algorithm_completed",
                algorithm=algorithm,
                variant_id=variant_id,
                target_transform=target_transform,
                metrics=result.overall_metrics,
            )
        except Exception as exc:
            logger.exception("Algorithm failed: %s", algorithm)
            failed.append(variant_id)
            rows.append(
                {
                    "algorithm": variant_id,
                    "base_algorithm": algorithm,
                    "target_transform": target_transform,
                    "label": display_label,
                    "status": "failed",
                    "is_baseline": False,
                    "error": str(exc),
                }
            )
            context.tracker.event(
                "algorithm_failed",
                algorithm=algorithm,
                variant_id=variant_id,
                target_transform=target_transform,
                error=str(exc),
            )
            print(
                f"  ✗ {display_label} "
                f"falló: {exc}"
            )

    ranking = _rank(rows)
    ranking.to_csv(context.tables / "competition_ranking.csv", index=False)
    successful = ranking[
        (ranking["status"] == "ok")
        & (ranking["is_baseline"] != True)  # noqa: E712
    ].copy()

    if successful.empty:
        export_workbook(
            context.exports / "model_1_results.xlsx",
            _excel_sheets(prepared, ranking, results, baseline),
        )
        build_dashboard(context.root, "LithiumScope — Modelo 1")
        context.tracker.set_state(
            RunState.FAILED,
            failure="No trainable algorithm completed successfully.",
        )
        return CompetitionOutcome(context.root, ranking, None, [], failed)

    save_competition_chart(
        successful,
        context.figures / "competition_rmse.png",
    )
    winner_variant = str(
        successful.iloc[0]["algorithm"]
    )
    winner_result, winner_prepared = (
        results[winner_variant]
    )
    winner_algorithm = (
        winner_result.algorithm
    )
    winner_transform = (
        winner_result.target_transform
    )
    final_estimator = build_pipeline(
        winner_algorithm,
        device,
        seed,
        winner_prepared.schema,
        winner_result.final_params,
        final_fit=True,
        target_transform=winner_transform,
    )

    print(
        f"\n🏆 Ganador provisional: "
        f"{winner_result.label}"
    )
    print("Entrenando ganador con todas las muestras disponibles...")
    final_estimator.fit(winner_prepared.x, winner_prepared.y)

    save_feature_importance(
        final_estimator,
        context.tables / "winner_feature_importance.csv",
        context.figures / "winner_feature_importance.png",
    )

    absolute_residuals = np.abs(
        winner_prepared.y.to_numpy() - winner_result.predictions
    )
    interval_q90 = float(np.quantile(absolute_residuals, 0.90))
    profile = applicability_profile(
        winner_prepared.x,
        winner_prepared.schema.numeric,
    )

    metadata = {
        "model_group": "model_1",
        "algorithm": winner_algorithm,
        "variant_id": winner_variant,
        "target_transform": winner_transform,
        "label": winner_result.label,
        "samples": len(winner_prepared.y),
        "metrics": winner_result.overall_metrics,
        "competition_primary_metric": "rmse_mean",
        "winner_rule": (
            "lowest mean outer-CV RMSE across algorithm/target variants; "
            "tie-break MAE then R2"
        ),
        "ranking": successful.to_dict(orient="records"),
        "run_id": context.run_id,
        "device": device.accelerator,
        "device_name": device.name,
        "validation_strategy": (
            str(
                winner_result.fold_table[
                    "validation_strategy"
                ].iloc[0]
            )
            if (
                not winner_result.fold_table.empty
                and "validation_strategy"
                in winner_result.fold_table.columns
            )
            else "unknown"
        ),
        "spatial_groups": (
            int(winner_prepared.groups.nunique())
            if winner_prepared.groups is not None
            else 0
        ),
        "spatial_group_degrees": config["validation"].get(
            "spatial_group_degrees"
        ),
        "oof_absolute_residual_q90": interval_q90,
        "dataset_sha256": dataset_manifest.source_sha256,
        "git_commit": context.tracker.payload.get("runtime", {}).get("git_commit"),
        "preflight_status": preflight.get("status"),
        "preflight_manifest": str(
            context.manifests / "model_1_preflight.json"
        ),
    }
    bundle = {
        "estimator": final_estimator,
        "algorithm": winner_algorithm,
        "variant_id": winner_variant,
        "target_transform": winner_transform,
        "target": winner_prepared.target,
        "schema": winner_prepared.schema,
        "oof_absolute_residual_q90": interval_q90,
        "applicability_profile": profile,
    }
    model_path, metadata_path = save_model_bundle(
        "model_1",
        f"winner_{winner_variant}",
        bundle,
        metadata,
        run_id=context.run_id,
        dataset_manifest_path=manifest_path,
        feature_schema=winner_prepared.schema,
        config_name="model_1",
        activate=False,
    )

    export_workbook(
        context.exports / "model_1_results.xlsx",
        _excel_sheets(prepared, ranking, results, baseline),
    )
    (context.exports / "winner.txt").write_text(
        (
            f"variant={winner_variant}\n"
            f"algorithm={winner_algorithm}\n"
            f"target_transform={winner_transform}\n"
            f"model={model_path}\n"
            f"metadata={metadata_path}\n"
        ),
        encoding="utf-8",
    )
    build_dashboard(
        context.root,
        "LithiumScope — Modelo 1 · Competencia",
    )

    final_state = RunState.PARTIAL if failed else RunState.COMPLETED
    context.tracker.set_state(
        final_state,
        winner=winner_variant,
        winner_algorithm=winner_algorithm,
        target_transform=winner_transform,
        primary_metric="rmse_mean",
        primary_metric_value=float(successful.iloc[0]["rmse_mean"]),
        model_path=str(model_path),
        failed_algorithms=failed,
    )
    return CompetitionOutcome(
        context.root,
        ranking,
        winner_variant,
        list(results),
        failed,
    )
