"""Use canonical scientific checks without fit, cache reconstruction or acquisition."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from lithiumscope.tools.inspection.development.inventory import file_record, local_path


def _config(root, group):
    from lithiumscope.core.config import load_config
    # Missing config must not trigger installation of packaged defaults.
    if not local_path(root, f"config/{group}.yaml").is_file():
        raise FileNotFoundError("Configuration unavailable")
    return load_config(group)


def curve_summary(plan):
    return {"partitions": len(plan), "ready": sum(p.status == "ready" for p in plan),
            "status_counts": dict(Counter(p.status for p in plan)),
            "min_train_rows": min(len(p.train) for p in plan),
            "max_train_rows": max(len(p.train) for p in plan)}


def inspect_m1(root: Path) -> dict:
    from lithiumscope.core.scientific.learning_curves import build_curve_plan
    from lithiumscope.core.scientific.spatial_audit import audit_spatial_folds
    from lithiumscope.datasets.management.sources.local import local_training_source
    from lithiumscope.model_1.evaluation.baseline import baseline_splits
    from lithiumscope.model_1.pipeline import prepare_training_data

    cfg = _config(root, "model_1")
    source = local_path(root, local_training_source("model_1"))
    prepared = prepare_training_data(source)
    splits, strategy = baseline_splits(prepared, cfg)
    distance, _ = audit_spatial_folds(
        prepared.frame, prepared.groups, splits,
        minimum_distance_m=float(cfg["validation"].get("minimum_distance_m", 0)),
    )
    plan = build_curve_plan(prepared.y, prepared.groups, splits, kind="regression")
    from lithiumscope.core.scientific.nested_protocol import audit_nested_development, nested_summary
    nested = audit_nested_development(prepared.frame, prepared.groups, prepared.y, splits,
        inner_folds=int(cfg["validation"]["inner_folds"]),
        minimum_distance_m=float(cfg["validation"].get("minimum_distance_m", 0)),
        optimization=bool(cfg["optimization"]["enabled"]))
    return {"state": "OK" if distance.distance_policy_pass.all() and nested_summary(nested)["state"] != "blocked" else "BLOCKED",
            "source": file_record(root, source), "raw_rows": len(prepared.raw_frame),
            "prepared_rows": len(prepared.x), "predictors": len(prepared.x.columns),
            "predictor_names": list(prepared.x.columns),
            "missing_by_predictor": {str(k): int(v) for k, v in prepared.x.isna().sum().items()},
            "groups": int(prepared.groups.nunique()), "validation": strategy,
            "target_ppm_quantiles": {str(k): float(v) for k, v in prepared.y.quantile([0, .25, .5, .75, .95, .99, 1]).items()},
            "stages": [{k: r[k] for k in ("stage", "rows", "columns", "missing_cells",
                       "rows_excluded", "frame_digest_sha256")} for r in prepared.audit.records],
            "exclusion_counts": dict(Counter(r["reason"] for r in prepared.audit.exclusions)),
            "spatial_folds": distance.to_dict("records"),
            "curve_preflight": curve_summary(plan), "nested_spatial_preflight": nested_summary(nested),
            "nested_spatial_folds": nested.to_dict("records"), "fit_performed": False}


def inspect_m2(root: Path) -> dict:
    import numpy as np
    import pandas as pd
    from lithiumscope.core.scientific.learning_curves import build_curve_plan
    from lithiumscope.core.scientific.spatial_audit import audit_spatial_folds
    from lithiumscope.datasets.management.sources.local import local_training_source
    from lithiumscope.datasets.management.paths import read_relocation, resolve_image_path
    from lithiumscope.model_2.data.feature_cache import validate_manifest
    from lithiumscope.model_2.validation.configuration import validate_canonical_config
    from lithiumscope.model_2.validation.quality import training_quality_mask
    from lithiumscope.model_2.validation.spatial import make_scientific_folds, scientific_labels
    from lithiumscope.model_2.preparation.features.v4_inputs import (
        combine_v4_quality_mask, spectral_v4_paths, verify_spectral_v4_source,
    )

    cfg = _config(root, "model_2")
    effective = validate_canonical_config(cfg)
    if cfg["training"].get("expansion", {}).get("enabled"):
        return {"state": "BLOCKED", "reason": "Temporary report covers frozen 706 cohort only"}
    source = local_path(root, local_training_source("model_2"))
    frame = pd.read_csv(source)
    validate_manifest(frame, cfg)
    target = cfg["training"]["lithium_column"]
    group = cfg["training"]["spatial_group_column"]
    result = {"state": "OK", "source": file_record(root, source), "rows": len(frame),
              "groups": int(frame[group].nunique()), "configuration": effective["effective"],
              "configuration_sha256": effective["configuration_sha256"], "fit_performed": False}
    relocation_path = local_path(root, "data/processed/model_2/relocation.json")
    relocation = read_relocation(root) if relocation_path.is_file() else {}
    missing = []
    for position, raw in enumerate(frame["image_path"]):
        try:
            path = local_path(root, resolve_image_path(raw, root=root, relocation=relocation))
            if not path.is_file() or not path.stat().st_size:
                missing.append(position)
        except (ValueError, OSError):
            missing.append(position)
    result["base_patches"] = {"available": len(frame)-len(missing), "missing_or_outside": len(missing),
                              "first_missing_positions": missing[:20]}
    if missing:
        result.update(state="BLOCKED", reason="Base patches unavailable; no acquisition attempted")
        return result
    keep = np.ones(len(frame), dtype=bool)
    quality = cfg["quality_training"]
    if quality["enabled"]:
        inventory = local_path(root, quality["inventory_path"])
        keep, _ = training_quality_mask(source, inventory_path=inventory,
                                        minimum_joint_valid=quality["minimum_joint_valid_fraction"])
    result["sixband_training_eligible"] = int(keep.sum())
    if cfg["training"].get("spectral_v4", {}).get("enabled"):
        matrix, state, qa = spectral_v4_paths(source, cfg)
        for path in (matrix, state, qa):
            local_path(root, path)
        # Ensure offline red-edge inputs cannot resolve outside this checkout.
        local_path(root, "results/scientific_simulations/m2_rededge_v4_20261113/patches")
        matrix, evidence = verify_spectral_v4_source(source, cfg)
        data = pd.read_csv(matrix)
        if (data.shape != (len(frame), 172) or not data.columns.is_unique
                or list(data.columns[:2]) != [target, group]
                or not np.array_equal(data[target].to_numpy(float), frame[target].to_numpy(float))
                or not np.array_equal(data[group].astype(str), frame[group].astype(str))
                or not np.isfinite(data.iloc[:, 2:].to_numpy(float)).all()):
            raise ValueError("V4 feature/label contract mismatch")
        keep, _ = combine_v4_quality_mask(keep, source, cfg)
        result["v4"] = {"matrix": file_record(root, matrix), "features": 170,
                        "raw_spectral_evidence": evidence["raw_spectral_evidence"],
                        "labels_and_groups_aligned": True, "base_126_reextracted": False}
    model = cfg["model"]
    validation = cfg["validation"]
    folds = make_scientific_folds(frame[target], frame[group], n_splits=validation["folds"],
                                  random_state=model["random_seed"], quantile=model["high_lithium_quantile"],
                                  fixed_threshold_ppm=model.get("fixed_threshold_ppm"))
    scientific_labels(folds, len(frame))
    splits = [(f.train_idx, f.test_idx) for f in folds]
    distance, _ = audit_spatial_folds(frame, frame[group], splits, eligible=keep,
                                     patch_side_m=cfg["imagery"]["patch_size_m"],
                                     minimum_distance_m=float(validation.get("minimum_distance_m", 0)))
    endpoints = [f.label_threshold_ppm for f in folds]
    plan = build_curve_plan(frame[target], frame[group], splits, kind="classification",
                            eligible=keep, endpoints=endpoints)
    supports = [{"fold": i, "train_positive": int(f.y_train[keep[f.train_idx]].sum()),
                 "train_negative": int((1-f.y_train[keep[f.train_idx]]).sum()),
                 "test_positive": int(f.y_test.sum()), "test_negative": int((1-f.y_test).sum()),
                 "endpoint_ppm": f.label_threshold_ppm} for i, f in enumerate(folds, 1)]
    from lithiumscope.core.scientific.nested_protocol import audit_nested_development, nested_summary
    opt = cfg["optimization"]
    final_endpoint = model.get("fixed_threshold_ppm")
    if final_endpoint is None:
        final_endpoint = float(frame[target].quantile(model["high_lithium_quantile"]))
    nested = audit_nested_development(frame, frame[group], frame[target], splits,
        inner_folds=validation["inner_folds"], eligible=keep, endpoints=endpoints, final_endpoint=final_endpoint,
        quantile=model["high_lithium_quantile"], fixed_endpoint=model.get("fixed_threshold_ppm"),
        minimum_distance_m=float(validation.get("minimum_distance_m", 0)), patch_side_m=cfg["imagery"]["patch_size_m"],
        optimization=bool(opt["enabled"]), operating=True,
        calibration=opt.get("calibration_policy") == "spatial_sigmoid" or any(
            name in {"svm_rbf", "random_forest_spatial_calibrated"} for name in model["algorithms"]))
    result.update(nested_spatial_preflight=nested_summary(nested), nested_spatial_folds=nested.to_dict("records"),
                  training_eligible=int(keep.sum()), all_test_retained=True,
                  spatial_folds=distance.to_dict("records"), class_support=supports,
                  curve_preflight=curve_summary(plan))
    if not distance.distance_policy_pass.all() or nested_summary(nested)["state"] == "blocked" or any(min(s["train_positive"], s["train_negative"]) == 0 for s in supports):
        result["state"] = "BLOCKED"
    return result
