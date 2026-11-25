"""Extracted responsibility; original public entrypoint remains stable."""

from __future__ import annotations


def _finalize_scientific_winner(
    baseline,
    candidates,
    cfg,
    context,
    dataset_manifest,
    dataset_manifest_path,
    device,
    failed,
    fixed_threshold,
    frame,
    groups,
    inner_folds,
    q,
    quality_audit,
    quality_cfg,
    quality_mask,
    ranking,
    results,
    seed,
    threshold,
    x,
    y,
    y_full,
):
    from lithiumscope.model_2.training.competition import (
        CompetitionOutcome,
        FEATURE_EXTRACTOR_VERSION,
        RunState,
        _excel_sheets,
        build_dashboard,
        export_workbook,
        feature_range_profile,
        get_label,
        np,
        pd,
        save_model_bundle,
        save_oof_diagnostics,
    )

    winner = str(candidates.iloc[0]["algorithm"])
    print(f"\n🏆 Ganador de esta ejecución Modelo 2: {get_label(winner)}")

    winner_result = results[winner]
    diagnostic = save_oof_diagnostics(
        context.root / "scientific_diagnostics",
        y.to_numpy(),
        winner_result.probabilities,
        groups=groups,
        kind="classification",
        seed=seed,
    )
    context.tracker.attach_summary(scientific_diagnostics=diagnostic["metrics"])
    threshold_cfg = cfg.get("decision_threshold", {})
    threshold_objective = str(threshold_cfg.get("objective", "balanced_accuracy"))
    from lithiumscope.model_2.training.calibration.policy import calibration_policy
    from lithiumscope.model_2.training.calibration.operating import learn_operating_policy, fit_policy

    final_fit_idx = np.flatnonzero(quality_mask) if quality_mask is not None else np.arange(len(x), dtype=int)
    x_final = x.iloc[final_fit_idx]
    y_final = y_full.iloc[final_fit_idx]
    groups_final = groups.iloc[final_fit_idx]
    selected_calibration = calibration_policy(cfg.get("optimization", {}))
    def fit_final_policy(a,b,c):
        return fit_policy(winner, device, seed, winner_result.final_params, a,b,c,inner_folds,selected_calibration)
    # Learn the final cut on development-only cross-fits of the selected family,
    # at the actual final Li endpoint. No outer-test-selected scores are reused.
    frozen, final_inner = learn_operating_policy(
        x_final, y_final, groups_final, fit=fit_final_policy,
        row_indices=final_fit_idx, reserved_indices=[], n_splits=inner_folds,
        objective=threshold_objective)
    threshold_selection = frozen.selection
    operating_threshold = float(threshold_selection.threshold)
    final_inner.to_csv(context.tables / "winner_operating_development_oof.csv", index=False)
    pd.DataFrame([{
        "threshold": operating_threshold, "objective": threshold_objective,
        "objective_value": threshold_selection.objective_value,
        "candidate_count": threshold_selection.candidate_count,
        "scope": "development_selection_not_external_evaluation",
        "label_threshold_ppm": threshold,
        "inner_rows_sha256": frozen.inner_rows_sha256,
        **{f"metric_{key}": value for key,value in threshold_selection.metrics.items()},
    }]).to_csv(context.tables / "winner_operating_threshold.csv", index=False)
    print(f"Umbral operativo de desarrollo: {operating_threshold:.4f}")
    from lithiumscope.persistence.final_fit_contract import freeze_final_fit
    from lithiumscope.core.pipeline_audit import stable_source_row_id
    fit_contract=freeze_final_fit(x=x_final,y=y_final,groups=groups_final,
        row_ids=[stable_source_row_id(dataset_manifest.source_sha256,int(idx)) for idx in final_fit_idx],
        dataset_sha256=dataset_manifest.source_sha256,
        training_signature=context.tracker.payload.get("summary",{}).get("training_signature"),
        params=winner_result.final_params,
        policy={"algorithm":winner,"li_cutoff_ppm":threshold,"operating_threshold":operating_threshold,
                "calibration":selected_calibration,"quality_policy":quality_audit})
    final_estimator = fit_final_policy(x_final,y_final,groups_final)
    profile = feature_range_profile(x_final)

    baseline_row = baseline[0]
    roc_gain = float(candidates.iloc[0]["roc_auc_mean"] - baseline_row["roc_auc_mean"])
    ap_gain = float(candidates.iloc[0]["average_precision_mean"] - baseline_row["average_precision_mean"])
    gate = cfg.get("release_gate", {})
    min_roc_gain = float(gate.get("min_roc_auc_gain_over_baseline", 0.0))
    min_ap_gain = float(
        gate.get(
            "min_average_precision_gain_over_baseline",
            0.0,
        )
    )
    # Outer OOF selects the family; final grouped development scores select the cut.
    # It cannot certify a release without a separate, unconsulted external holdout.
    exploratory_gate_pass = roc_gain >= min_roc_gain and ap_gain >= min_ap_gain
    release_gate_pass = False
    print(
        "Release gate Modelo 2: "
        + ("CANDIDATO / NO VERIFICADO" if exploratory_gate_pass else "FAIL")
        + (f" | ΔROC-AUC={roc_gain:+.4f} (mín {min_roc_gain:+.4f}) | ΔAP={ap_gain:+.4f} (mín {min_ap_gain:+.4f})")
    )

    from lithiumscope.model_2.preparation.features.v4_inputs import spectral_v4_enabled

    v4 = spectral_v4_enabled(cfg)
    extractor_version = 4 if v4 else FEATURE_EXTRACTOR_VERSION
    representation = "historical_126_plus_real_S2_rededge_44" if v4 else "legacy_DN_median_126"
    metadata = {
        "model_group": "model_2",
        "algorithm": winner,
        "label": get_label(winner),
        "samples": len(frame),
        "training_samples_after_quality": len(x_final),
        "train_only_quality": quality_audit,
        "experimental_only": True,
        "final_fit_contract_sha256":fit_contract["identity_sha256"],
        "eligible_for_individual_release": False,
        "random_forest_search_profile": cfg.get("optimization", {}).get("random_forest_search_profile", "legacy"),
        "calibration_policy": selected_calibration,
        "spectral_treatment":winner_result.final_params.get("spectral_treatment","legacy"),
        "provisional_georoc_sample_count": max(0, len(frame) - 706),
        "threshold_ppm": threshold,
        "target_quantile": (None if fixed_threshold is not None else q),
        "threshold_source": (
            "fixed_threshold_ppm" if fixed_threshold is not None else "dataset_quantile_final_fit_only"
        ),
        "metrics": winner_result.overall_metrics,
        "feature_extractor_version": extractor_version,
        "spectral_representation": representation,
        "real_rededge_same_scene_required": v4,
        "rededge_band_assets": (["rededge1", "rededge2", "rededge3", "nir08"] if v4 else []),
        "model1_chemistry_used_as_m2_features": False,
        "operating_threshold": operating_threshold,
        "operating_threshold_objective": threshold_selection.objective,
        "operating_threshold_objective_value": threshold_selection.objective_value,
        "operating_threshold_metrics": threshold_selection.metrics,
        "operating_threshold_scope": "development_selection_not_external_evaluation",
        "competition_primary_metric": "roc_auc_mean",
        "winner_rule": ("highest mean ROC-AUC; tie-break Average Precision then Balanced Accuracy"),
        "ranking": candidates.to_dict(orient="records"),
        "run_id": context.run_id,
        "device": device.accelerator,
        "dataset_sha256": dataset_manifest.source_sha256,
        "release_gate": {
            "pass": release_gate_pass,
            "exploratory_gate_pass": exploratory_gate_pass,
            "reason": "no_independent_external_holdout",
            "roc_auc_gain_over_baseline": roc_gain,
            "average_precision_gain_over_baseline": ap_gain,
            "min_roc_auc_gain_over_baseline": min_roc_gain,
            "min_average_precision_gain_over_baseline": min_ap_gain,
        },
        "cv_label_protocol": "group_kfold_fold_training_only",
        "operating_threshold_validity": "grouped_development_oof_not_external_holdout",
        "scope_note": ("Score de priorización exploratoria; no es probabilidad de yacimiento económicamente viable."),
    }

    from lithiumscope.model_2.preparation.patch_contract import make_patch_contract
    patch_contract = make_patch_contract(bands=cfg["imagery"]["bands"],
        pixels=cfg["imagery"].get("patch_pixels",32), size_m=cfg["imagery"].get("patch_size_m",640))
    metadata["patch_input_contract"] = patch_contract
    bundle = {
        "patch_input_contract": patch_contract,
        "estimator": final_estimator,
        "final_fit_contract":fit_contract,
        "features": list(x.columns),
        "band_names": list(cfg["imagery"]["bands"]),
        "normalize_per_band": bool(cfg["imagery"].get("normalize_per_band", False)),
        "threshold_ppm": threshold,
        "target_quantile": (None if fixed_threshold is not None else q),
        "algorithm": winner,
        "feature_extractor_version": extractor_version,
        "rededge_same_scene_required": v4,
        "rededge_band_assets": (["rededge1", "rededge2", "rededge3", "nir08"] if v4 else []),
        "operating_threshold": operating_threshold,
        "operating_threshold_objective": threshold_selection.objective,
        "operating_threshold_metrics": threshold_selection.metrics,
        "operating_threshold_scope": "development_selection_not_external_evaluation",
        "applicability_profile": profile,
        "experimental_only": True,
        "calibration_policy": selected_calibration,
        "quality_min_joint_valid": (
            float(quality_cfg.get("minimum_joint_valid_fraction", 0.95)) if quality_mask is not None else None
        ),
        "spectral_representation": representation,
        "spectral_treatment":winner_result.final_params.get("spectral_treatment","legacy"),
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
        activate=False,
        verification_probe=x_final.iloc[np.unique(np.linspace(0,len(x_final)-1,min(32,len(x_final)),dtype=int))],
        config_snapshot_path=context.config_snapshot/"model_2.yaml",
    )

    export_workbook(
        context.exports / "model_2_results.xlsx",
        _excel_sheets(ranking, results, y, baseline, context.tables),
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
        exploratory_gate_pass=exploratory_gate_pass,
        release_gate_reason="no_independent_external_holdout",
        roc_auc_gain_over_baseline=roc_gain,
        average_precision_gain_over_baseline=ap_gain,
        operating_threshold=operating_threshold,
        operating_threshold_objective=threshold_selection.objective,
        operating_threshold_objective_value=threshold_selection.objective_value,
    )
    return CompetitionOutcome(
        context.root,
        ranking,
        winner,
        candidates.algorithm.astype(str).tolist(),
        failed,
    )
