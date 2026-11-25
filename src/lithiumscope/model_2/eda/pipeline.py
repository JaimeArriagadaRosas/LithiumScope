"""Model 2 EDA uses the configured training feature loader, not M1 preparation."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from lithiumscope.core.config import load_config
from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.paths import RESULTS_DIR
from lithiumscope.core.scientific.eda_statistics import cohort_summary, feature_profiles, numeric_correlations
from lithiumscope.model_2.pipeline import load_model_2_training_frame
from lithiumscope.model_2.eda.quality import patch_quality
from lithiumscope.core.scientific.fold_profiles import fold_profiles
from lithiumscope.core.scientific.spatial_audit import audit_spatial_folds
from lithiumscope.model_2.validation.spatial import make_scientific_folds
from lithiumscope.model_2.preparation.features.v4_inputs import spectral_v4_enabled, load_spectral_v4_training_frame
from lithiumscope.results.eda_artifacts import complete_run, fail_run, find_compatible, new_run, signature
from lithiumscope.results.eda_figures import save_figures
from lithiumscope.results.eda_protocol import fold_protocol, attach_training_eda


def run_eda(path: Path, *, output_root: Path | None = None, frame=None, config=None,
            scientific_folds=None, quality_mask=None) -> Path:
    cfg = load_config("model_2") if config is None else config
    training = cfg["training"]
    target = str(training["lithium_column"])
    group = str(training.get("spatial_group_column", "spatial_group"))
    manifest = pd.read_csv(path)
    if frame is None:
        frame = (load_spectral_v4_training_frame(path, cfg) if spectral_v4_enabled(cfg)
                 else load_model_2_training_frame(path))
    frame = frame.reset_index(drop=True)
    if (len(frame) != len(manifest) or not np.array_equal(frame[target].to_numpy(), manifest[target].to_numpy())
            or group not in frame or group not in manifest
            or not np.array_equal(frame[group].astype(str).to_numpy(), manifest[group].astype(str).to_numpy())):
        raise ValueError("M2 EDA feature rows do not match manifest labels/groups")
    features = [name for name in frame if name not in {target, group, "sample_id"}]
    metadata = [name for name in ("sample_id", "source_dataset", "longitude", "latitude",
                "sentinel_scene_id", "sentinel_cloud_cover", "sentinel_datetime") if name in manifest]
    combined = frame.reset_index(drop=True).copy()
    for name in metadata:
        if name not in combined:
            combined[name] = manifest[name].to_numpy()
    qa, qa_audit = patch_quality(path, cfg, manifest)
    for name in qa:
        if name not in combined:
            combined[name] = qa[name].to_numpy()
    if "source_cohort" in combined and "source_cohort" not in metadata:
        metadata.append("source_cohort")
    root = output_root if output_root is not None else RESULTS_DIR / "model_2" / "eda"
    validation = cfg.get("validation", {})
    folds, protocol, fold_state = scientific_folds, None, "not_available"
    eligibility = qa["train_eligible"].to_numpy(bool) if "train_eligible" in qa else None
    if quality_mask is not None and (eligibility is None or not np.array_equal(eligibility, quality_mask)):
        raise ValueError("EDA and training QA disagree")
    if folds is None and "folds" in validation and combined[group].nunique() >= int(validation["folds"]):
        model = cfg["model"]
        try:
            folds = make_scientific_folds(combined[target], combined[group],
                n_splits=int(validation["folds"]), random_state=int(model.get("random_seed", 42)),
                quantile=float(model["high_lithium_quantile"]),
                fixed_threshold_ppm=model.get("fixed_threshold_ppm"))
        except ValueError:
            fold_state = "blocked_spatial_partition"
    if folds is not None:
        protocol = fold_protocol(frame[features], combined[target], combined[group],
            [(f.train_idx, f.test_idx) for f in folds], eligible=eligibility,
            endpoints=[f.label_threshold_ppm for f in folds])
    key = signature("model_2", path, cfg, combined, protocol=protocol)
    cached = find_compatible(root, key)
    if cached:
        print(f"[OK] EDA descriptivo compatible reutilizado: {cached}")
        return cached
    directory = new_run(root)
    try:
        tables = directory / "tables"
        profiles = feature_profiles(combined, features, stage="prepared")
        profiles.to_csv(tables / "feature_profiles.csv", index=False)
        cohort_summary(combined, target).to_csv(tables / "cohort_summary.csv", index=False)
        combined[[target, group, *metadata]].to_csv(tables / "sample_provenance.csv", index=False)
        if "sentinel_scene_id" in combined:
            scene = combined.groupby("sentinel_scene_id", dropna=False)[target].agg(["count", "median"])
            if "sentinel_cloud_cover" in combined:
                scene["scene_cloud_mean"] = pd.to_numeric(
                    combined["sentinel_cloud_cover"], errors="coerce").groupby(combined["sentinel_scene_id"]).mean()
            scene.to_csv(tables / "scene_quality.csv")
        if not qa.empty:
            qa.to_csv(tables / "patch_quality.csv", index=False)
        if folds is not None:
            fold_summary, fold_features = fold_profiles(frame[features], combined[target], combined[group],
                [(f.train_idx, f.test_idx) for f in folds], eligibility)
            fold_summary["label_threshold_ppm"] = [f.label_threshold_ppm for f in folds]
            fold_summary["train_positive_count"] = [
                int(f.y_train[eligibility[f.train_idx]].sum()) if eligibility is not None else int(f.y_train.sum())
                for f in folds]
            fold_summary.to_csv(tables / "outer_train_summary.csv", index=False)
            fold_features.to_csv(tables / "outer_train_features.csv", index=False)
            spatial_summary, spatial_rows = audit_spatial_folds(combined, combined[group],
                [(f.train_idx, f.test_idx) for f in folds], eligible=eligibility,
                patch_side_m=float(cfg["imagery"].get("patch_size_m", 640)),
                minimum_distance_m=float(validation.get("minimum_distance_m", 0)))
            spatial_summary.to_csv(tables / "outer_spatial_summary.csv", index=False)
            spatial_rows.to_csv(tables / "outer_spatial_nearest.csv", index=False)
            fold_state = "group_kfold_train_profiles"
        correlations = numeric_correlations(combined, [*features, target])
        correlations.to_csv(tables / "spearman_correlations.csv")
        save_figures(directory / "figures", combined, target, profiles, correlations)
        print(cohort_summary(combined, target).to_string(index=False))
        return complete_run(directory, signature_value=key, details={
            "model_group": "model_2", "source_path": str(path), "source_sha256": file_sha256(path),
            "rows_raw": len(manifest), "rows_prepared": len(frame), "features": len(features),
            "feature_names": features, "effective_config": cfg,
            "quality_audit": qa_audit, "fold_profile_state": fold_state, "development_protocol": protocol,
            "feature_extractor": "configured_v4" if spectral_v4_enabled(cfg) else "configured_legacy",
            "limitations": ["Full-cohort descriptive analysis, not holdout evidence.",
                            "V4 local re-extraction verifies all 44 red-edge descriptors; legacy block verified separately." if spectral_v4_enabled(cfg) else "Legacy cache bound to ordered sample and image identities.",
                            "Scene cloud cover is not patch-local valid-pixel quality."],
        })
    except Exception as exc:
        fail_run(directory, exc)
        raise


def record_training_eda(path, frame, config, folds, quality_mask, context):
    target = config["training"]["lithium_column"]
    group = config["training"]["spatial_group_column"]
    x = frame.drop(columns=[target, group])
    protocol = fold_protocol(x, frame[target], frame[group],
        [(f.train_idx, f.test_idx) for f in folds], eligible=quality_mask,
        endpoints=[f.label_threshold_ppm for f in folds])
    directory = run_eda(path, frame=frame, config=config,
                        scientific_folds=folds, quality_mask=quality_mask)
    attach_training_eda(directory, context, protocol)
    context.tracker.event("eda_evidence_verified", eda_run_id=directory.name)
