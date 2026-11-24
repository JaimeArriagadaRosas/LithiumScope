"""Extracted responsibility; original public entrypoint remains stable."""
from __future__ import annotations


def _prepare_scientific_manifests(context, fixed_threshold, frame, groups, inner_folds, manifest_path, q, scientific_folds, threshold):
    from lithiumscope.model_2.training.competition import build_tabular_manifest, evaluate_scientific_baseline, write_manifest
    dataset_manifest = build_tabular_manifest(
        manifest_path,
        dataset_name="model_2_training_manifest",
        model_group="model_2",
        stage="spectral_feature_training_input",
        frame=frame,
        metadata={
            "threshold_ppm": threshold,
            "target_quantile": (
                None
                if fixed_threshold is not None
                else q
            ),
            "threshold_source": (
                "fixed_threshold_ppm"
                if fixed_threshold is not None
                else "dataset_quantile_final_fit_only"
            ),
            "cv_label_source": "outer_train_quantile_only",
            "cv_split": "group_kfold_strict",
            "spatial_groups": int(groups.nunique()) if groups is not None else 0,
        },
    )
    dataset_manifest_path = write_manifest(
        dataset_manifest,
        context.manifests / "dataset_manifest.json",
    )
    context.tracker.attach_summary(dataset_sha256=dataset_manifest.source_sha256)

    folds = len(scientific_folds)
    threshold_source = (
        "fijo"
        if fixed_threshold is not None
        else f"cuantil {q:.2f}"
    )
    print(
        f"Muestras: {len(frame)} | Umbral Li alto: "
        f"{threshold:.4f} ppm ({threshold_source}) | "
        f"Folds: {folds}"
    )
    if groups is not None and groups.nunique() >= folds:
        print(
            f"Validación espacial: {groups.nunique()} grupos "
            f"| inner folds: {inner_folds}"
        )

    baseline = evaluate_scientific_baseline(scientific_folds, len(frame))
    baseline[1].to_csv(
        context.tables / "baseline_fold_metrics.csv",
        index=False,
    )

    return baseline, dataset_manifest, dataset_manifest_path, folds
