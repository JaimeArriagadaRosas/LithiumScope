from __future__ import annotations

from lithiumscope.core.scientific.candidate_evidence import eligible_rows

from pathlib import Path
import json

import pandas as pd

from lithiumscope.core.config import load_config
from lithiumscope.core.paths import PROJECT_ROOT
from lithiumscope.resources.detection.device import DeviceInfo
from lithiumscope.core.logging.session import get_logger
from lithiumscope.core.reproducibility import set_global_seed
from lithiumscope.core.run_context import RunContext
from lithiumscope.core.states import RunState
from lithiumscope.model_2.evaluation.plots import save_competition_chart
from lithiumscope.model_2.training.postprocess import finalize_reports
from lithiumscope.results.dashboard import build_dashboard
from lithiumscope.results.excel_exporter import export_workbook
from lithiumscope.runtime.graceful_shutdown import get_shutdown_manager

from lithiumscope.model_2.training.execution.contracts import CompetitionOutcome, CompetitionOptions, CandidateEvidence
from lithiumscope.model_2.evaluation.artifacts import competition_workbook_sheets
from lithiumscope.model_2.evaluation.selection import rank_candidates

from lithiumscope.model_2.training.execution.scientific_data import prepare_scientific_data
from lithiumscope.model_2.training.execution.scientific_manifests import prepare_scientific_manifests
from lithiumscope.model_2.training.execution.candidate_evaluation import evaluate_candidates
from lithiumscope.model_2.training.execution.winner_finalization import finalize_winner

logger = get_logger("model_2.competition")


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
    from lithiumscope.model_2.validation.configuration import validate_canonical_config
    effective_config=validate_canonical_config(cfg)
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
    optimization_cfg = dict(cfg.get("optimization", {}))
    decision = cfg.get("decision_threshold", {})
    if decision.get("policy") == "grouped_development_oof":
        optimization_cfg["operating_threshold_objective"] = decision.get("objective", "balanced_accuracy")
    else:
        raise ValueError("Canonical M2 requires grouped development operating thresholds")
    inner_folds = int(cfg["validation"].get("inner_folds", 3))

    from lithiumscope.model_2.preparation.features.v4_inputs import (
        build_scientific_training_signature,
    )
    signature = build_scientific_training_signature(manifest_path, cfg)
    context = RunContext.create(
        "model_2",
        training_signature=signature,
        resume=True,
    )

    print("\n=== MODELO 2 · COMPETENCIA DE ALGORITMOS ===")
    print(f"Entrenamiento: {context.run_id}")
    logger.info("M2 competition started: %s", context.run_id)

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
    (context.manifests/"effective_configuration.json").write_text(json.dumps(effective_config,indent=2),encoding="utf-8")
    get_shutdown_manager().register_cleanup(context.tracker.cancel_if_active)

    data = prepare_scientific_data(cfg, context, fixed_threshold, lithium_column, manifest_path, q, requested_folds, seed, spatial_group_column)
    options = CompetitionOptions(fixed_threshold, inner_folds, lithium_column, manifest_path, optimization_cfg, q, seed)

    from lithiumscope.model_2.eda.pipeline import record_training_eda
    record_training_eda(manifest_path, data.frame, cfg, data.scientific_folds, data.quality_mask, context)
    manifests = prepare_scientific_manifests(context, data, options)
    baseline = manifests.baseline
    y = data.y

    rows: list[dict] = [baseline[0]]
    results: dict[str, object] = {}
    failed: list[str] = []

    evidence = CandidateEvidence(results, rows, failed)
    evaluate_candidates(algorithms, context, device, data, manifests, options, evidence)

    ranking, candidates = rank_candidates(rows)
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
        logger.error("No eligible M2 candidate: %s", context.run_id)
        export_workbook(
            context.exports / "model_2_results.xlsx",
            competition_workbook_sheets(ranking, results, y, baseline, context.tables),
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

    return finalize_winner(cfg, context, device, data, manifests, options, evidence, ranking, candidates)
