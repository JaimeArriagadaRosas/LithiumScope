from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from lithiumscope.core.config import load_config
from lithiumscope.core.device import DeviceInfo
from lithiumscope.core.logger import get_logger
from lithiumscope.core.reproducibility import set_global_seed
from lithiumscope.core.run_context import RunContext
from lithiumscope.core.run_resume import build_training_signature
from lithiumscope.core.states import RunState
from lithiumscope.datasets.manifest import build_tabular_manifest, write_manifest
from lithiumscope.model_2.evaluation.baseline import evaluate_prior_baseline
from lithiumscope.model_2.evaluation.maps import save_spatial_score_map
from lithiumscope.model_2.evaluation.plots import (
    save_competition_chart,
    save_probability_histogram,
    save_roc_pr,
)
from lithiumscope.model_2.pipeline import load_model_2_training_frame
from lithiumscope.model_2.schema import feature_range_profile
from lithiumscope.model_2.training.cv_runner import run_classification_cv
from lithiumscope.model_2.training.factory import create_model, get_label
from lithiumscope.persistence.save_model import save_model_bundle
from lithiumscope.results.dashboard import build_dashboard
from lithiumscope.results.excel_exporter import export_workbook
from lithiumscope.runtime.graceful_shutdown import get_shutdown_manager

logger = get_logger("model_2.competition")


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
        "algorithm": result.algorithm,
        "label": result.label,
        "roc_auc_mean": float(folds["roc_auc"].mean()),
        "roc_auc_std": float(folds["roc_auc"].std(ddof=0)),
        "average_precision_mean": float(folds["average_precision"].mean()),
        "balanced_accuracy_mean": float(folds["balanced_accuracy"].mean()),
        "f1_mean": float(folds["f1"].mean()),
        "precision_mean": float(folds["precision"].mean()),
        "recall_mean": float(folds["recall"].mean()),
        "brier_score_mean": float(folds["brier_score"].mean()),
        "elapsed_seconds": result.elapsed_seconds,
        "status": "ok",
        "is_baseline": False,
    }


def _excel_sheets(
    ranking: pd.DataFrame,
    results: dict[str, object],
    y: pd.Series,
    baseline,
) -> dict[str, pd.DataFrame]:
    sheets: dict[str, pd.DataFrame] = {
        "ranking": ranking,
        "baseline_folds": baseline[1],
    }
    for algorithm, result in results.items():
        sheets[f"folds_{algorithm}"[:31]] = result.fold_table
        sheets[f"pred_{algorithm}"[:31]] = pd.DataFrame(
            {
                "y_true": y.to_numpy(),
                "prospectivity_score": result.probabilities,
                "predicted_class_0_5": (result.probabilities >= 0.5).astype(int),
            }
        )
    return sheets


def run_model_2_competition(
    manifest_path: Path,
    device: DeviceInfo,
) -> CompetitionOutcome:
    cfg = load_config("model_2")
    app_cfg = load_config("app")
    algorithms = list(cfg["model"]["algorithms"])
    seed = int(cfg["model"]["random_seed"])
    set_global_seed(
        seed,
        deterministic_torch=bool(
            app_cfg.get("reproducibility", {}).get("deterministic_torch", False)
        ),
    )

    lithium_column = str(cfg["training"]["lithium_column"])
    spatial_group_column = str(
        cfg["training"].get("spatial_group_column", "spatial_group")
    )
    requested_folds = int(cfg["validation"].get("folds", 5))
    q = float(cfg["model"]["high_lithium_quantile"])
    fixed_threshold = cfg["model"].get("fixed_threshold_ppm")
    optimization_cfg = cfg.get("optimization", {})
    inner_folds = int(cfg["validation"].get("inner_folds", 3))

    signature = build_training_signature(
        "model_2",
        manifest_path,
        ("app", "model_2"),
    )
    context = RunContext.create(
        "model_2",
        training_signature=signature,
        resume=True,
    )

    print("\n=== MODELO 2 · COMPETENCIA DE ALGORITMOS ===")
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

    frame = load_model_2_training_frame(manifest_path)
    threshold = (
        float(fixed_threshold)
        if fixed_threshold is not None
        else float(frame[lithium_column].quantile(q))
    )
    y = (frame[lithium_column] >= threshold).astype(int)
    groups = (
        frame[spatial_group_column]
        if spatial_group_column in frame.columns
        else None
    )
    x = frame.drop(
        columns=[
            column
            for column in (lithium_column, spatial_group_column)
            if column in frame.columns
        ]
    )

    dataset_manifest = build_tabular_manifest(
        manifest_path,
        dataset_name="model_2_training_manifest",
        model_group="model_2",
        stage="spectral_feature_training_input",
        frame=frame,
        metadata={
            "threshold_ppm": threshold,
            "target_quantile": (
                None
                if fixed_threshold is not None
                else q
            ),
            "threshold_source": (
                "fixed_threshold_ppm"
                if fixed_threshold is not None
                else "dataset_quantile"
            ),
            "spatial_groups": int(groups.nunique()) if groups is not None else 0,
        },
    )
    dataset_manifest_path = write_manifest(
        dataset_manifest,
        context.manifests / "dataset_manifest.json",
    )
    context.tracker.attach_summary(dataset_sha256=dataset_manifest.source_sha256)

    min_class = int(y.value_counts().min())
    folds = max(2, min(requested_folds, min_class))
    threshold_source = (
        "fijo"
        if fixed_threshold is not None
        else f"cuantil {q:.2f}"
    )
    print(
        f"Muestras: {len(frame)} | Umbral Li alto: "
        f"{threshold:.4f} ppm ({threshold_source}) | "
        f"Folds: {folds}"
    )
    if groups is not None and groups.nunique() >= folds:
        print(
            f"Validación espacial: {groups.nunique()} grupos "
            f"| inner folds: {inner_folds}"
        )

    baseline = evaluate_prior_baseline(
        x=x,
        y=y,
        folds=folds,
        seed=seed,
        groups=groups,
    )
    baseline[1].to_csv(
        context.tables / "baseline_fold_metrics.csv",
        index=False,
    )

    rows: list[dict] = [baseline[0]]
    results: dict[str, object] = {}
    failed: list[str] = []

    for index, algorithm in enumerate(algorithms, start=1):
        print(f"\n[{index}/{len(algorithms)}] Entrenando {get_label(algorithm)}")
        context.tracker.event("algorithm_started", algorithm=algorithm)

        try:
            result = run_classification_cv(
                x,
                y,
                algorithm,
                device,
                seed,
                folds,
                groups,
                checkpoint_root=context.checkpoints,
                optimization=optimization_cfg,
                inner_folds=inner_folds,
            )
            results[algorithm] = result
            rows.append(_ranking_row(result))

            result.fold_table.to_csv(
                context.tables / f"fold_metrics_{algorithm}.csv",
                index=False,
            )
            pd.DataFrame(
                {
                    "y_true": y,
                    "prospectivity_score": result.probabilities,
                    "predicted_class_0_5": (
                        result.probabilities >= 0.5
                    ).astype(int),
                }
            ).to_csv(
                context.tables / f"oof_predictions_{algorithm}.csv",
                index=False,
            )

            save_probability_histogram(
                y,
                result.probabilities,
                context.figures / algorithm / "score_distribution.png",
                f"{result.label} — Distribución de scores",
            )
            save_roc_pr(
                y,
                result.probabilities,
                context.figures / algorithm / "roc_pr.png",
                result.label,
            )
            save_spatial_score_map(
                manifest_path,
                result.probabilities,
                context.figures / algorithm / "spatial_scores.png",
                f"{result.label} — scores espaciales OOF",
            )
            context.tracker.event(
                "algorithm_completed",
                algorithm=algorithm,
                metrics=result.overall_metrics,
            )
        except Exception as exc:
            logger.exception("Model 2 algorithm failed: %s", algorithm)
            failed.append(algorithm)
            rows.append(
                {
                    "algorithm": algorithm,
                    "label": get_label(algorithm),
                    "status": "failed",
                    "is_baseline": False,
                    "error": str(exc),
                }
            )
            context.tracker.event(
                "algorithm_failed",
                algorithm=algorithm,
                error=str(exc),
            )
            print(f"  ✗ {get_label(algorithm)} falló: {exc}")

    ranking = pd.DataFrame(rows)
    candidates = ranking[
        (ranking["status"] == "ok")
        & (ranking["is_baseline"] != True)  # noqa: E712
    ].copy()
    baseline_rows = ranking[ranking["is_baseline"] == True].copy()  # noqa: E712
    failed_rows = ranking[ranking["status"] != "ok"].copy()

    if not candidates.empty:
        candidates = candidates.sort_values(
            [
                "roc_auc_mean",
                "average_precision_mean",
                "balanced_accuracy_mean",
            ],
            ascending=[False, False, False],
        ).reset_index(drop=True)
        candidates.insert(0, "rank", range(1, len(candidates) + 1))
    if not baseline_rows.empty:
        baseline_rows.insert(0, "rank", pd.NA)

    ranking = pd.concat(
        [candidates, baseline_rows, failed_rows],
        ignore_index=True,
        sort=False,
    )
    ranking.to_csv(
        context.tables / "competition_ranking.csv",
        index=False,
    )

    if candidates.empty:
        export_workbook(
            context.exports / "model_2_results.xlsx",
            _excel_sheets(ranking, results, y, baseline),
        )
        build_dashboard(context.root, "LithiumScope — Modelo 2")
        context.tracker.set_state(
            RunState.FAILED,
            failure="No trainable algorithm completed successfully.",
        )
        return CompetitionOutcome(context.root, ranking, None, [], failed)

    save_competition_chart(
        candidates,
        context.figures / "competition_roc_auc.png",
    )

    winner = str(candidates.iloc[0]["algorithm"])
    print(f"\n🏆 Ganador de esta ejecución Modelo 2: {get_label(winner)}")

    winner_result = results[winner]
    final_calibration_cv = None
    if winner == "svm_rbf":
        from lithiumscope.model_2.training.cv_runner import _materialize_splits

        final_calibration_cv = _materialize_splits(
            x,
            y,
            groups,
            inner_folds,
            seed + 9000,
        )

    final_estimator = create_model(
        winner,
        device,
        seed,
        winner_result.final_params,
        calibrated=True,
        calibration_cv=final_calibration_cv,
    )
    final_estimator.fit(x, y)
    profile = feature_range_profile(x)

    baseline_row = baseline[0]
    roc_gain = float(
        candidates.iloc[0]["roc_auc_mean"]
        - baseline_row["roc_auc_mean"]
    )
    ap_gain = float(
        candidates.iloc[0]["average_precision_mean"]
        - baseline_row["average_precision_mean"]
    )
    gate = cfg.get("release_gate", {})
    min_roc_gain = float(
        gate.get("min_roc_auc_gain_over_baseline", 0.0)
    )
    min_ap_gain = float(
        gate.get(
            "min_average_precision_gain_over_baseline",
            0.0,
        )
    )
    release_gate_pass = (
        roc_gain >= min_roc_gain
        and ap_gain >= min_ap_gain
    )

    metadata = {
        "model_group": "model_2",
        "algorithm": winner,
        "label": get_label(winner),
        "samples": len(frame),
        "threshold_ppm": threshold,
        "target_quantile": (
            None if fixed_threshold is not None else q
        ),
        "threshold_source": (
            "fixed_threshold_ppm"
            if fixed_threshold is not None
            else "dataset_quantile"
        ),
        "metrics": winner_result.overall_metrics,
        "competition_primary_metric": "roc_auc_mean",
        "winner_rule": (
            "highest mean ROC-AUC; tie-break Average Precision then "
            "Balanced Accuracy"
        ),
        "ranking": candidates.to_dict(orient="records"),
        "run_id": context.run_id,
        "device": device.accelerator,
        "dataset_sha256": dataset_manifest.source_sha256,
        "release_gate": {
            "pass": release_gate_pass,
            "roc_auc_gain_over_baseline": roc_gain,
            "average_precision_gain_over_baseline": ap_gain,
            "min_roc_auc_gain_over_baseline": min_roc_gain,
            "min_average_precision_gain_over_baseline": min_ap_gain,
        },
        "scope_note": (
            "Score de priorización exploratoria; no es probabilidad de "
            "yacimiento económicamente viable."
        ),
    }

    bundle = {
        "estimator": final_estimator,
        "features": list(x.columns),
        "band_names": list(cfg["imagery"]["bands"]),
        "normalize_per_band": bool(
            cfg["imagery"].get("normalize_per_band", False)
        ),
        "threshold_ppm": threshold,
        "target_quantile": (
            None if fixed_threshold is not None else q
        ),
        "algorithm": winner,
        "applicability_profile": profile,
    }
    model_path, metadata_path = save_model_bundle(
        "model_2",
        f"winner_{winner}",
        bundle,
        metadata,
        run_id=context.run_id,
        dataset_manifest_path=dataset_manifest_path,
        feature_schema={"features": list(x.columns)},
        config_name="model_2",
    )

    export_workbook(
        context.exports / "model_2_results.xlsx",
        _excel_sheets(ranking, results, y, baseline),
    )
    (context.exports / "winner.txt").write_text(
        f"algorithm={winner}\nmodel={model_path}\nmetadata={metadata_path}\n",
        encoding="utf-8",
    )

    build_dashboard(
        context.root,
        "LithiumScope — Modelo 2 · Competencia",
    )

    final_state = RunState.PARTIAL if failed else RunState.COMPLETED
    context.tracker.set_state(
        final_state,
        winner=winner,
        primary_metric="roc_auc_mean",
        primary_metric_value=float(candidates.iloc[0]["roc_auc_mean"]),
        model_path=str(model_path),
        failed_algorithms=failed,
        release_gate_pass=release_gate_pass,
        roc_auc_gain_over_baseline=roc_gain,
        average_precision_gain_over_baseline=ap_gain,
    )
    return CompetitionOutcome(
        context.root,
        ranking,
        winner,
        list(results),
        failed,
    )
