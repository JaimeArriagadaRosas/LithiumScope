"""Extracted responsibility; original public entrypoint remains stable."""
from __future__ import annotations


def _finalize_model_1_winner(baseline, config, context, dataset_manifest, device, failed, manifest_path, preflight, prepared, results, rows, seed):
    from lithiumscope.model_1.training.competition import CompetitionOutcome, RunState, _excel_sheets, _rank, applicability_profile, build_dashboard, build_pipeline, export_workbook, np, save_competition_chart, save_feature_importance, save_model_bundle, save_oof_diagnostics
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

    diagnostic = save_oof_diagnostics(
        context.root / "scientific_diagnostics",
        winner_prepared.y.to_numpy(), winner_result.predictions,
        groups=winner_prepared.groups, kind="regression", seed=seed,
    )
    context.tracker.attach_summary(scientific_diagnostics=diagnostic["metrics"])
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
