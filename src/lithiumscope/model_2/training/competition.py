from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd

from lithiumscope.core.config import load_config
from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.paths import PROJECT_ROOT
from lithiumscope.core.device import DeviceInfo
from lithiumscope.core.logger import get_logger
from lithiumscope.core.reproducibility import set_global_seed
from lithiumscope.core.run_context import RunContext
from lithiumscope.core.training_diagnostics import save_oof_diagnostics
from lithiumscope.core.run_resume import build_training_signature
from lithiumscope.core.states import RunState
from lithiumscope.datasets.manifest import build_tabular_manifest, write_manifest
from lithiumscope.model_2.evaluation.baseline import evaluate_prior_baseline
from lithiumscope.model_2.evaluation.thresholds import select_operating_threshold
from lithiumscope.model_2.evaluation.maps import save_spatial_score_map
from lithiumscope.model_2.evaluation.plots import (
    save_competition_chart,
    save_probability_histogram,
    save_roc_pr,
)
from lithiumscope.model_2.pipeline import load_model_2_training_frame
from lithiumscope.model_2.steps.step_05_spectral_features import (
    FEATURE_EXTRACTOR_VERSION,
)
from lithiumscope.model_2.schema import feature_range_profile
from lithiumscope.model_2.training.cv_runner import run_classification_cv
from lithiumscope.model_2.training.quality import training_quality_mask
from lithiumscope.model_2.training.postprocess import finalize_reports
from lithiumscope.model_2.training.scientific_validation import (
    make_scientific_folds, make_anchored_scientific_folds,
    scientific_labels, evaluate_scientific_baseline,
)
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
    evidence_dir: Path | None = None,
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
    if evidence_dir is not None:
        for filename,sheet in (
            ("old_release_protocol_comparison_706.csv","release_706"),
            ("old_release_protocol_intervals_706.csv","release_intervals"),
            ("scientific_stratified_metrics.csv","scientific_strata"),
            ("three_zone_competition_cases.csv","three_zone_cases"),
        ):
            evidence_path=evidence_dir/filename
            if evidence_path.exists():
                sheets[sheet]=pd.read_csv(evidence_path)
    return sheets


def run_model_2_competition(
    manifest_path: Path,
    device: DeviceInfo,
) -> CompetitionOutcome:
    cfg = load_config("model_2")
    if not cfg["training"].get("expansion", {}).get("enabled", False):
        frozen_path = PROJECT_ROOT / str(cfg["training"]["manifest_path"])
        if Path(manifest_path).resolve() != frozen_path.resolve():
            raise ValueError(
                "M2 historical-only mode requires the frozen 706-row manifest"
            )
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

    fold_ids, frame, groups, quality_audit, quality_cfg, quality_mask, scientific_folds, threshold, x, y, y_full = _prepare_scientific_data(cfg, context, fixed_threshold, lithium_column, manifest_path, q, requested_folds, seed, spatial_group_column)

    baseline, dataset_manifest, dataset_manifest_path, folds = _prepare_scientific_manifests(context, fixed_threshold, frame, groups, inner_folds, manifest_path, q, scientific_folds, threshold)

    rows: list[dict] = [baseline[0]]
    results: dict[str, object] = {}
    failed: list[str] = []

    _evaluate_algorithm_candidates(algorithms, context, device, failed, fixed_threshold, fold_ids, folds, frame, groups, inner_folds, lithium_column, manifest_path, optimization_cfg, q, quality_mask, results, rows, scientific_folds, seed, x, y)

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
    if results:
        # All three reports are repairable from saved OOF without retraining.
        # Preserve a long-running competition even when one report needs repair,
        # but never hide the reason or mark the report as successful.
        evidence_status=finalize_reports(
            manifest_path,context.tables,context.tables,
        )
        per_report=evidence_status.get("reports",{})
        context.tracker.event(
            "scientific_postprocessing_complete",
            status=evidence_status["state"],
            report_statuses={
                name:details.get("state")
                for name,details in per_report.items()
            },
            recovery_command=evidence_status.get("recovery_command"),
        )
        triple=per_report.get("three_zone",{})
        if triple.get("state")=="complete":
            arms=triple["result"]["summary"]["arms"]
            ranking["three_zone_top1_accuracy"]=ranking["algorithm"].map({
                name:float(value["top1_accuracy"])
                for name,value in arms.items()
            })
            print("Priorizacion de tres zonas por algoritmo:",flush=True)
            for name,value in arms.items():
                print(f"  {name}: {value['top1_accuracy']:.2%}",flush=True)
        if evidence_status["state"]!="complete":
            print(
                "ATENCION: evidencias complementarias incompletas; "
                "los modelos y checkpoints quedan conservados. "
                "Consultar evidence_report_status.json para recuperar "
                "SIN reentrenar.",flush=True
            )
        else:
            print("Reportes del release previo, QA cientifica y tres zonas: OK",
                  flush=True)
    ranking.to_csv(
        context.tables / "competition_ranking.csv",
        index=False,
    )

    if candidates.empty:
        export_workbook(
            context.exports / "model_2_results.xlsx",
            _excel_sheets(ranking, results, y, baseline, context.tables),
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

    return _finalize_scientific_winner(baseline, candidates, cfg, context, dataset_manifest, dataset_manifest_path, device, failed, fixed_threshold, frame, groups, inner_folds, q, quality_audit, quality_cfg, quality_mask, ranking, results, seed, threshold, x, y, y_full)

from lithiumscope.model_2.training.competition_prepare_scientific_data import _prepare_scientific_data
from lithiumscope.model_2.training.competition_prepare_scientific_manifests import _prepare_scientific_manifests
from lithiumscope.model_2.training.competition_evaluate_algorithm_candidates import _evaluate_algorithm_candidates
from lithiumscope.model_2.training.competition_finalize_scientific_winner import _finalize_scientific_winner
