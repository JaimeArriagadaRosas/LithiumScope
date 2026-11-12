"""Extracted responsibility; original public entrypoint remains stable."""
from __future__ import annotations


def _evaluate_algorithm_candidates(algorithms, context, device, failed, fixed_threshold, fold_ids, folds, frame, groups, inner_folds, lithium_column, manifest_path, optimization_cfg, q, quality_mask, results, rows, scientific_folds, seed, x, y):
    from lithiumscope.model_2.training.competition import _ranking_row, get_label, logger, pd, run_classification_cv, save_probability_histogram, save_roc_pr, save_spatial_score_map
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
                scientific_folds=scientific_folds,
                lithium_values=frame[lithium_column].to_numpy(dtype=float),
                label_quantile=q,
                fixed_threshold_ppm=fixed_threshold,
                quality_training_mask=quality_mask,
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
                    "li_ppm": frame[lithium_column].to_numpy(dtype=float),
                    "outer_fold": fold_ids.to_numpy(),
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
