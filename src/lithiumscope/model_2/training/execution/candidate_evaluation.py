"""Independent M2 scientific training component."""
from __future__ import annotations
from lithiumscope.model_2.training.execution.contracts import ScientificData, ScientificManifests, CompetitionOptions, CandidateEvidence


def evaluate_candidates(algorithms, context, device, data: ScientificData, manifests: ScientificManifests,
                        options: CompetitionOptions, evidence: CandidateEvidence):
    fold_ids = data.fold_ids
    frame = data.frame
    groups = data.groups
    quality_mask = data.quality_mask
    scientific_folds = data.scientific_folds
    x = data.x
    y = data.y
    folds = manifests.folds
    fixed_threshold = options.fixed_threshold
    inner_folds = options.inner_folds
    lithium_column = options.lithium_column
    manifest_path = options.manifest_path
    optimization_cfg = options.optimization_cfg
    q = options.q
    seed = options.seed
    failed = evidence.failed
    results = evidence.results
    rows = evidence.rows
    from lithiumscope.model_2.evaluation.selection import ranking_row
    from lithiumscope.model_2.training.factory import get_label
    from lithiumscope.core.logging.session import get_logger
    logger = get_logger("model_2.candidate_evaluation")
    import pandas as pd
    from lithiumscope.model_2.training.execution.cv_runner import run_classification_cv
    from lithiumscope.model_2.evaluation.plots import save_probability_histogram
    from lithiumscope.model_2.evaluation.plots import save_roc_pr
    from lithiumscope.model_2.evaluation.maps import save_spatial_score_map
    from lithiumscope.core.scientific.candidate_evidence import record_evidence, verify_candidate
    from lithiumscope.core.scientific.classification_metrics import validate_binary_inputs, reliability_table
    from lithiumscope.model_2.evaluation.oof_evidence import build_oof_table
    from lithiumscope.resources.allocation.competition_workers import schedule_candidates, candidate_result, enabled
    from lithiumscope.resources.allocation.estimates import estimate_fit_demand
    from lithiumscope.resources.manager import load_policy
    scheduled=None
    policy=load_policy()
    if enabled(device,policy):
        requests=[]
        for algorithm in algorithms:
            demand=estimate_fit_demand(x,algorithm,folds=int(folds),inner_folds=int(inner_folds),
                trials=int(optimization_cfg.get("per_algorithm_trials",{}).get(algorithm,optimization_cfg.get("n_trials",8))),
                optimize=bool(optimization_cfg.get("enabled",False)))
            requests.append((algorithm,{"x":x,"y":y,"algorithm":algorithm,"device":device,"seed":seed,
                "folds":folds,"groups":groups,"checkpoint_root":context.checkpoints,"optimization":optimization_cfg,
                "inner_folds":inner_folds,"scientific_folds":scientific_folds,
                "lithium_values":frame[lithium_column].to_numpy(dtype=float),"label_quantile":q,
                "fixed_threshold_ppm":fixed_threshold,"quality_training_mask":quality_mask},demand.working_memory_estimate_bytes))
        scheduled=schedule_candidates("model_2",requests,context.checkpoints,device,policy=policy)
    for index, algorithm in enumerate(algorithms, start=1):
        print(f"\n[{index}/{len(algorithms)}] Entrenando {get_label(algorithm)}")
        context.tracker.event("algorithm_started", algorithm=algorithm)

        try:
            from lithiumscope.resources.allocation.estimates import estimate_fit_demand
            from lithiumscope.resources.manager import fit_workload_scope
            demand=estimate_fit_demand(x,algorithm,folds=int(folds),inner_folds=int(inner_folds),
                trials=int(optimization_cfg.get("per_algorithm_trials",{}).get(algorithm,optimization_cfg.get("n_trials",8))),
                optimize=bool(optimization_cfg.get("enabled",False)))
            if scheduled is not None:
                result=candidate_result(scheduled,algorithm)
            else:
                with fit_workload_scope(demand,evidence_path=context.manifests/f"resource_plan_{algorithm}.json"):
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
                    scientific_folds=scientific_folds,
                    lithium_values=frame[lithium_column].to_numpy(dtype=float),
                    label_quantile=q,
                    fixed_threshold_ppm=fixed_threshold,
                    quality_training_mask=quality_mask,
                )
            validate_binary_inputs(y, result.probabilities)
            verify_candidate(result, result.probabilities, len(y), ("roc_auc", "average_precision", "balanced_accuracy"), folds)
            row = ranking_row(result)
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
            continue

        results[algorithm] = result

        def write_predictions():
            table = build_oof_table(pd.read_csv(manifest_path), frame[lithium_column], y, fold_ids,
                                    result.probabilities, scientific_folds, quality_mask)
            if optimization_cfg.get("operating_threshold_objective"):
                cuts = result.fold_table.set_index("fold")["operating_threshold"]
                table["operating_threshold"] = table.outer_fold.map(cuts)
                if table.operating_threshold.isna().any():
                    raise ValueError("Missing fold operating thresholds")
                table["predicted_class_operating"] = (table.prospectivity_score >= table.operating_threshold).astype(int)
                result.inner_operating_predictions.to_csv(context.tables / f"inner_operating_{algorithm}.csv", index=False)
            table.to_csv(context.tables / f"oof_predictions_{algorithm}.csv", index=False)

        record_evidence(
            row, context.tracker, logger, algorithm,
            {
                "fold_metrics": lambda: result.fold_table.to_csv(context.tables / f"fold_metrics_{algorithm}.csv", index=False),
                "oof": write_predictions,
                "reliability": lambda: reliability_table(y, result.probabilities).to_csv(
                    context.tables / f"reliability_{algorithm}.csv", index=False),
            },
            {
                "histogram": lambda: save_probability_histogram(y, result.probabilities,
                    context.figures / algorithm / "score_distribution.png", f"{result.label} - Distribucion"),
                "roc_pr": lambda: save_roc_pr(y, result.probabilities,
                    context.figures / algorithm / "roc_pr.png", result.label),
                "map": lambda: save_spatial_score_map(manifest_path,
                    pd.read_csv(context.tables / f"oof_predictions_{algorithm}.csv"),
                    context.figures / algorithm / "spatial_scores.png", f"{result.label} - OOF"),
            },
        )
        rows.append(row)
        context.tracker.event("algorithm_completed", algorithm=algorithm, metrics=result.overall_metrics)
