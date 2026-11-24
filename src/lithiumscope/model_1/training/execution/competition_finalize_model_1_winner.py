"""Extracted responsibility; original public entrypoint remains stable."""
from __future__ import annotations

from lithiumscope.core.scientific.candidate_evidence import eligible_rows


def _finalize_model_1_winner(baseline, config, context, dataset_manifest, device, failed, manifest_path, preflight, prepared, results, rows, seed):
    from lithiumscope.model_1.training.competition import CompetitionOutcome, RunState, _excel_sheets, _rank, applicability_profile, build_dashboard, build_pipeline, export_workbook, np, save_competition_chart, save_feature_importance, save_model_bundle, save_oof_diagnostics
    from lithiumscope.model_1.evaluation.selection import baseline_eligibility
    ranking = baseline_eligibility(_rank(rows), preflight_ok=preflight.get("status") == "pass",
        minimum_gain=float(config.get("baseline",{}).get("minimum_rmse_gain_ppm",0.0)))
    ranking.to_csv(context.tables / "competition_ranking.csv", index=False)
    successful = ranking[
        eligible_rows(ranking)
        & (ranking["is_baseline"] != True)  # noqa: E712
    ].copy()

    relative_winner = str(successful.iloc[0]["algorithm"]) if not successful.empty else None
    eligible = successful.loc[successful.scientific_candidate_eligible]
    if eligible.empty:
        export_workbook(
            context.exports / "model_1_results.xlsx",
            _excel_sheets(prepared, ranking, results, baseline),
        )
        build_dashboard(context.root, "LithiumScope — Modelo 1")
        context.tracker.set_state(
            RunState.PARTIAL if relative_winner else RunState.FAILED,
            relative_winner=relative_winner,
            failure="No candidate satisfies predeclared baseline and preflight criteria.",
        )
        return CompetitionOutcome(context.root, ranking, None, successful.algorithm.astype(str).tolist(), failed)

    save_competition_chart(
        successful,
        context.figures / "competition_rmse.png",
    )
    winner_variant = str(
        eligible.iloc[0]["algorithm"]
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
    from lithiumscope.model_1.training.preprocessing.spatial_tabnet import fit_regression_estimator, early_stopping_evidence
    from lithiumscope.persistence.final_fit_contract import freeze_final_fit
    from lithiumscope.core.pipeline_audit import stable_source_row_id
    fit_contract=freeze_final_fit(x=winner_prepared.x,y=winner_prepared.y,groups=winner_prepared.groups,
        row_ids=[stable_source_row_id(dataset_manifest.source_sha256,idx) for idx in winner_prepared.x.index],
        dataset_sha256=dataset_manifest.source_sha256,
        training_signature=context.tracker.payload.get("summary",{}).get("training_signature"),
        params=winner_result.final_params,
        policy={"algorithm":winner_algorithm,"target_transform":winner_transform,
                "schema":vars(winner_prepared.schema),"scope":"development_only"})
    fit_regression_estimator(final_estimator,winner_prepared.x,winner_prepared.y,winner_prepared.groups)
    tabnet_evidence=early_stopping_evidence(final_estimator)

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
        "tabnet_early_stopping": tabnet_evidence,
        "relative_winner": relative_winner,
        "baseline_policy": str(eligible.iloc[0]["baseline_policy"]),
        "rmse_gain_over_best_baseline":float(eligible.iloc[0]["rmse_gain_over_best_baseline"]),
        "experimental_only":True,
        "final_fit_contract_sha256":fit_contract["identity_sha256"],
        "release_gate":{"pass":False,"reason":"no_independent_external_holdout"},
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
        "final_fit_contract":fit_contract,
        "experimental_only":True,
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
        verification_probe=winner_prepared.x.iloc[np.unique(np.linspace(0,len(winner_prepared.x)-1,min(32,len(winner_prepared.x)),dtype=int))],
        config_snapshot_path=context.config_snapshot/"model_1.yaml",
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

    final_state = RunState.PARTIAL if failed or len(successful)!=len(results) else RunState.COMPLETED
    context.tracker.set_state(
        final_state,
        relative_winner=relative_winner,
        release_gate_pass=False,
        winner=winner_variant,
        winner_algorithm=winner_algorithm,
        target_transform=winner_transform,
        primary_metric="rmse_mean",
        primary_metric_value=float(eligible.iloc[0]["rmse_mean"]),
        model_path=str(model_path),
        failed_algorithms=failed,
    )
    return CompetitionOutcome(
        context.root,
        ranking,
        winner_variant,
        successful.algorithm.astype(str).tolist(),
        failed,
    )
