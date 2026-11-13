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
        create_model,
        export_workbook,
        feature_range_profile,
        get_label,
        np,
        pd,
        save_model_bundle,
        save_oof_diagnostics,
        select_operating_threshold,
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
    threshold_fallback = float(threshold_cfg.get("fallback", 0.5))
    # Threshold is a deployable candidate only. Its OOF objective is optimistic
    # because model selection and threshold choice reuse the same OOF pool.
    threshold_selection = select_operating_threshold(
        y.to_numpy(),
        winner_result.probabilities,
        objective=threshold_objective,
        fallback=threshold_fallback,
    )
    operating_threshold = float(threshold_selection.threshold)
    print(
        "Umbral operativo Modelo 2: "
        f"{operating_threshold:.4f} "
        f"| {threshold_objective} OOF="
        f"{threshold_selection.objective_value:.4f}"
    )
    pd.DataFrame(
        [
            {
                "threshold": operating_threshold,
                "objective": threshold_selection.objective,
                "objective_value": threshold_selection.objective_value,
                "candidate_count": threshold_selection.candidate_count,
                **{f"metric_{key}": value for key, value in threshold_selection.metrics.items()},
            }
        ]
    ).to_csv(
        context.tables / "winner_operating_threshold.csv",
        index=False,
    )

    final_fit_idx = np.flatnonzero(quality_mask) if quality_mask is not None else np.arange(len(x), dtype=int)
    x_final = x.iloc[final_fit_idx]
    y_final = y_full.iloc[final_fit_idx]
    groups_final = groups.iloc[final_fit_idx] if groups is not None else None
    if y_final.nunique() != 2:
        raise ValueError("M2 quality TRAIN has only one class")
    final_calibration_cv = None
    if winner in {"svm_rbf", "random_forest_spatial_calibrated"}:
        from sklearn.model_selection import GroupKFold

        if groups_final is None or groups_final.nunique() < inner_folds:
            raise ValueError("No spatial groups for final calibration.")
        final_calibration_cv = list(
            GroupKFold(n_splits=inner_folds).split(
                x_final,
                groups=groups_final,
            )
        )
        for train_idx, valid_idx in final_calibration_cv:
            if y_final.iloc[train_idx].nunique() < 2 or y_final.iloc[valid_idx].nunique() < 2:
                raise ValueError("Final calibration requires both classes per spatial fold.")

    from lithiumscope.model_2.training.calibration_policy import (
        calibration_policy,
        fit_spatially_calibrated,
    )

    selected_calibration = calibration_policy(cfg.get("optimization", {}))
    if selected_calibration == "spatial_sigmoid":
        final_estimator = fit_spatially_calibrated(
            winner,
            device,
            seed,
            winner_result.final_params,
            x_final,
            y_final,
            groups_final,
            inner_folds,
        )
    else:
        final_estimator = create_model(
            winner,
            device,
            seed,
            winner_result.final_params,
            calibrated=True,
            calibration_cv=final_calibration_cv,
        )
        final_estimator.fit(x_final, y_final)
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
    # OOF selected both the winning candidate and an operating threshold.
    # It cannot certify a release without a separate, unconsulted external holdout.
    exploratory_gate_pass = roc_gain >= min_roc_gain and ap_gain >= min_ap_gain
    release_gate_pass = False
    print(
        "Release gate Modelo 2: "
        + ("CANDIDATO / NO VERIFICADO" if exploratory_gate_pass else "FAIL")
        + (f" | ΔROC-AUC={roc_gain:+.4f} (mín {min_roc_gain:+.4f}) | ΔAP={ap_gain:+.4f} (mín {min_ap_gain:+.4f})")
    )

    from lithiumscope.model_2.training.v4_inputs import spectral_v4_enabled

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
        "eligible_for_individual_release": False,
        "random_forest_search_profile": cfg.get("optimization", {}).get("random_forest_search_profile", "legacy"),
        "calibration_policy": selected_calibration,
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
        "operating_threshold_validity": "OOF_selection_exploratory_not_holdout",
        "scope_note": ("Score de priorización exploratoria; no es probabilidad de yacimiento económicamente viable."),
    }

    bundle = {
        "estimator": final_estimator,
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
        "applicability_profile": profile,
        "experimental_only": True,
        "calibration_policy": selected_calibration,
        "quality_min_joint_valid": (
            float(quality_cfg.get("minimum_joint_valid_fraction", 0.95)) if quality_mask is not None else None
        ),
        "spectral_representation": representation,
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
        list(results),
        failed,
    )
