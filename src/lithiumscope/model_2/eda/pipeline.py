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
from lithiumscope.model_2.training.v4_inputs import spectral_v4_enabled, load_spectral_v4_training_frame
from lithiumscope.results.eda_artifacts import complete_run, fail_run, find_compatible, new_run, signature
from lithiumscope.results.eda_figures import save_figures


def run_eda(path: Path, *, output_root: Path | None = None) -> Path:
    cfg = load_config("model_2")
    training = cfg["training"]
    target = str(training["lithium_column"])
    group = str(training.get("spatial_group_column", "spatial_group"))
    manifest = pd.read_csv(path)
    frame = (load_spectral_v4_training_frame(path, cfg) if spectral_v4_enabled(cfg)
             else load_model_2_training_frame(path))
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
    root = output_root if output_root is not None else RESULTS_DIR / "model_2" / "eda"
    key = signature("model_2", path, cfg, combined)
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
        correlations = numeric_correlations(combined, [*features, target])
        correlations.to_csv(tables / "spearman_correlations.csv")
        save_figures(directory / "figures", combined, target, profiles, correlations)
        print(cohort_summary(combined, target).to_string(index=False))
        return complete_run(directory, signature_value=key, details={
            "model_group": "model_2", "source_path": str(path), "source_sha256": file_sha256(path),
            "rows_raw": len(manifest), "rows_prepared": len(frame), "features": len(features),
            "feature_names": features, "effective_config": cfg,
            "feature_extractor": "configured_v4" if spectral_v4_enabled(cfg) else "configured_legacy",
            "limitations": ["Full-cohort descriptive analysis, not holdout evidence.",
                            "Legacy feature matrix alignment checked by labels/groups; row-level identity still pending.",
                            "Scene cloud cover is not patch-local valid-pixel quality."],
        })
    except Exception as exc:
        fail_run(directory, exc)
        raise
