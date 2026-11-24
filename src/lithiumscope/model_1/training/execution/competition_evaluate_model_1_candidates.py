"""Extracted responsibility; original public entrypoint remains stable."""
from __future__ import annotations


def _evaluate_model_1_candidates(config, context, device, experiments, failed, prepared, results, rows):
    from lithiumscope.model_1.training.competition import _ranking_row, get_algorithm, logger, run_nested_cv
    from lithiumscope.core.scientific.candidate_evidence import record_evidence, verify_candidate
    from lithiumscope.model_1.evaluation.artifacts import save_algorithm_tables, save_algorithm_figures
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
            verify_candidate(result, result.predictions, len(prepared.y), ("rmse", "mae", "r2"),
                expected_folds=len(prepared.outer_splits))
            from numpy import array_equal
            if result.outer_splits is None or len(result.outer_splits)!=len(prepared.outer_splits) or any(
                not (array_equal(a,c) and array_equal(b,d)) for (a,b),(c,d) in zip(result.outer_splits,prepared.outer_splits,strict=True)):
                raise ValueError("Candidate folds differ from frozen common partition")
            row = _ranking_row(result)
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
            continue

        results[variant_id] = (result, algorithm_prepared)
        record_evidence(
            row, context.tracker, logger, variant_id,
            {"tables": lambda: save_algorithm_tables(algorithm_prepared, result, context)},
            {"figures": lambda: save_algorithm_figures(algorithm_prepared, result, context)},
        )
        rows.append(row)
        context.tracker.event("algorithm_completed", algorithm=algorithm, variant_id=variant_id,
                              target_transform=target_transform, metrics=result.overall_metrics)
