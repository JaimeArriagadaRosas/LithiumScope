"""Extracted responsibility; original public entrypoint remains stable."""
from __future__ import annotations


def _evaluate_model_1_candidates(config, context, device, experiments, failed, prepared, results, rows):
    from lithiumscope.model_1.training.competition import _ranking_row, get_algorithm, logger, run_nested_cv, save_algorithm_artifacts
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
