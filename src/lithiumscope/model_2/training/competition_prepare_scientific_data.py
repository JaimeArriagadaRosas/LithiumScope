"""Extracted responsibility; original public entrypoint remains stable."""
from __future__ import annotations


def _prepare_scientific_data(cfg, context, fixed_threshold, lithium_column, manifest_path, q, requested_folds, seed, spatial_group_column):
    from lithiumscope.model_2.training.competition import PROJECT_ROOT, file_sha256, json, load_model_2_training_frame, make_anchored_scientific_folds, make_scientific_folds, np, pd, scientific_labels, training_quality_mask
    frame = load_model_2_training_frame(manifest_path)
    # Full-data target threshold is reserved for the final fitted model.
    # Evaluation labels are defined independently in each outer training fold.
    threshold = (
        float(fixed_threshold)
        if fixed_threshold is not None
        else float(frame[lithium_column].quantile(q))
    )
    groups = (
        frame[spatial_group_column]
        if spatial_group_column in frame.columns
        else None
    )
    x = frame.drop(
        columns=[
            column
            for column in (lithium_column, spatial_group_column)
            if column in frame.columns
        ]
    )
    quality_cfg=cfg.get("quality_training", {})
    quality_mask=None
    quality_audit={"enabled": False, "manifest_rows":len(frame),
                   "training_eligible":len(frame),"training_excluded":0}
    if bool(quality_cfg.get("enabled",False)):
        inventory=PROJECT_ROOT / str(quality_cfg["inventory_path"])
        quality_mask,quality_audit=training_quality_mask(
            manifest_path,inventory_path=inventory,
            minimum_joint_valid=float(
                quality_cfg.get("minimum_joint_valid_fraction",0.95)),
        )
        if len(quality_mask)!=len(frame):
            raise ValueError("Model 2 QA mask must align with all feature rows")
        (context.manifests/"train_only_quality_policy.json").write_text(
            json.dumps(quality_audit,indent=2,ensure_ascii=False),
            encoding="utf-8")
        print(
            "Calidad científica SOLO TRAIN: "
            f"{quality_audit['training_eligible']} / {len(frame)} "
            "observaciones elegibles, sin eliminar ninguna observación TEST.",
            flush=True
        )
    anchor = cfg["validation"].get("anchor_legacy", {})
    legacy_manifest = PROJECT_ROOT / str(cfg["training"]["manifest_path"])
    extended_manifest = manifest_path.resolve() != legacy_manifest.resolve()
    if extended_manifest and bool(anchor.get("enabled", False)):
        old_sha = str(cfg["training"]["frozen_manifest_sha256"])
        if file_sha256(legacy_manifest) != old_sha:
            raise RuntimeError("M2 frozen historical reference dataset changed")
        base = pd.read_csv(legacy_manifest)
        expanded = pd.read_csv(manifest_path)
        baseline_count = int(cfg["training"]["frozen_expected_rows"])
        if (len(base) != baseline_count
                or len(expanded) < baseline_count
                or not np.array_equal(
                    base.sample_id.to_numpy(),
                    expanded.sample_id.iloc[:baseline_count].to_numpy(),
                )
                or not np.allclose(
                    base.Li_icpms.to_numpy(dtype=float),
                    expanded.Li_icpms.iloc[:baseline_count].to_numpy(dtype=float),
                )
                or not np.array_equal(
                    base.spatial_group.to_numpy(),
                    expanded.spatial_group.iloc[:baseline_count].to_numpy(),
                )):
            raise RuntimeError("M2 expanded manifest altered historical sample rows")
        reference_file = PROJECT_ROOT / str(anchor["folds_file"])
        if file_sha256(reference_file) != str(anchor["folds_sha256"]):
            raise RuntimeError("M2 anchored baseline fold snapshot changed")
        historic_splits = json.loads(reference_file.read_text(encoding="utf-8"))
        scientific_folds = make_anchored_scientific_folds(
            frame[lithium_column], groups, historic_splits,
            baseline_count=baseline_count, quantile=q,
            fixed_threshold_ppm=fixed_threshold,
        )
        print(
            f"M2: evaluacion pareada con {baseline_count} muestras "
            "historicas en sus mismos folds espaciales."
        )
    else:
        scientific_folds = make_scientific_folds(
            frame[lithium_column], groups,
            n_splits=requested_folds, random_state=seed,
            quantile=q, fixed_threshold_ppm=fixed_threshold,
        )
    y = pd.Series(scientific_labels(scientific_folds, len(frame)), index=frame.index)
    y_full = (frame[lithium_column] >= threshold).astype(int)
    fold_ids = pd.Series(-1, index=frame.index, dtype=int)
    for fold_number, split in enumerate(scientific_folds, start=1):
        fold_ids.iloc[split.test_idx] = fold_number
    (context.manifests / "scientific_folds.json").write_text(
        __import__("json").dumps(
            [{
                "fold": i, "n_train": len(f.train_idx),
                "n_test": len(f.test_idx),
                "label_threshold_ppm": f.label_threshold_ppm,
                "n_groups_train": f.n_groups_train,
                "n_groups_test": f.n_groups_test,
                "train_indices": f.train_idx.tolist(),
                "test_indices": f.test_idx.tolist(),
            } for i, f in enumerate(scientific_folds, 1)],
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return fold_ids, frame, groups, quality_audit, quality_cfg, quality_mask, scientific_folds, threshold, x, y, y_full
