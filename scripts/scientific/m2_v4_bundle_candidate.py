"""Build a versioned, NOT ACTIVATED, 170-feature M2 winner bundle.

Requires all six complete v4 nested Optuna arms. Never touches active_models.
Inference must use the extra four STAC bands of the same original scene.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from lithiumscope.core.config import load_config
from lithiumscope.core.device import detect_device
from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.paths import PROJECT_ROOT
from lithiumscope.model_2.methodology.rededge_training import combine_quality_masks
from lithiumscope.model_2.training.calibration_policy import (
    fit_spatially_calibrated,
    calibration_policy,
)
from lithiumscope.model_2.training.factory import create_model
from lithiumscope.model_2.training.quality import training_quality_mask

from scripts.scientific.m2_rededge_v4_competition import ARMS


def train_final(*, root: Path = PROJECT_ROOT) -> dict:
    out = root / "results/scientific_simulations/m2_rededge_v4_20261113"
    nested = out / "nested_170_706"
    status_path = nested / "final_status.json"
    if not status_path.is_file():
        raise FileNotFoundError("All six nested competitors must complete")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("status") != "complete_experimental_no_release" or set(status.get("trained_algorithms", [])) != set(
        ARMS
    ):
        raise ValueError("Incomplete six-model M2 v4 nested competition")
    ranking = pd.read_csv(nested / "six_algorithm_nested_v4_vs_v3.csv")
    if set(ranking.algorithm) != set(ARMS):
        raise ValueError("M2 v4 release candidate ranking is incomplete")
    winner = (
        ranking.sort_values(
            ["roc_auc_mean", "v4_ap"],
            ascending=False,
        )
        .iloc[0]
        .algorithm
    )
    params = json.loads((nested / f"final_params_{winner}.json").read_text(encoding="utf-8"))
    source = out / "cohort_706/training_v4_170.csv"
    manifest_path = root / "data/processed/model_2/training_manifest.csv"
    manifest = pd.read_csv(manifest_path)
    data = pd.read_csv(source)
    if len(data) != 706 or len(data.columns) != 172:
        raise ValueError("M2 v4 source not completely acquired")
    x = data.drop(columns=["Li_icpms", "spatial_group"])
    if not np.array_equal(data.Li_icpms, manifest.Li_icpms):
        raise ValueError("M2 v4 candidate Li labels drifted")
    cfg = load_config("model_2")
    qa, _ = training_quality_mask(
        manifest_path,
        inventory_path=root / str(cfg["quality_training"]["inventory_path"]),
        minimum_joint_valid=float(cfg["quality_training"]["minimum_joint_valid_fraction"]),
    )
    qa, audit = combine_quality_masks(
        qa,
        out / "cohort_706/rededge_quality.csv",
        manifest,
    )
    fit_x = x.loc[qa]
    groups = manifest.spatial_group.loc[qa]
    threshold = float(
        np.quantile(
            manifest.Li_icpms.loc[qa].to_numpy(float),
            float(cfg["model"]["high_lithium_quantile"]),
        )
    )
    fit_y = (manifest.Li_icpms.loc[qa] >= threshold).astype(int)
    if set(fit_y.unique()) != {0, 1}:
        raise ValueError("M2 v4 final fit has only one class")
    device = detect_device(prefer_gpu=False)
    seed = int(cfg["model"]["random_seed"])
    if calibration_policy(cfg["optimization"]) == "spatial_sigmoid":
        model = fit_spatially_calibrated(
            winner,
            device,
            seed,
            params,
            fit_x,
            fit_y,
            groups,
            3,
        )
    else:
        cv = None
        if winner == "svm_rbf":
            cv = list(GroupKFold(3).split(fit_x, groups=groups))
            for t, v in cv:
                if len(set(fit_y.iloc[t])) != 2 or len(set(fit_y.iloc[v])) != 2:
                    raise ValueError("M2 v4 SVM calibrator lacks a class")
        model = create_model(
            winner,
            device,
            seed,
            params,
            calibrated=True,
            calibration_cv=cv,
        )
        model.fit(fit_x, fit_y)
    test_score = model.predict_proba(fit_x.iloc[:3])[:, 1]
    if not np.isfinite(test_score).all():
        raise ValueError("M2 v4 winner failed final smoke scoring")
    out_dir = out / "candidate_bundle_not_published"
    out_dir.mkdir(parents=True, exist_ok=True)
    active = root / "models/active_models.json"
    active_sha = file_sha256(active)
    bundle = {
        "estimator": model,
        "algorithm": winner,
        "features": list(x.columns),
        "feature_extractor_version": 4,
        "rededge_band_assets": ["rededge1", "rededge2", "rededge3", "nir08"],
        "rededge_same_scene_required": True,
        "base_band_names": ["red", "green", "blue", "nir", "swir16", "swir22"],
        "normalize_per_band": False,
        "threshold_ppm": threshold,
        "target_quantile": float(cfg["model"]["high_lithium_quantile"]),
        "experimental_only": True,
        "eligible_for_individual_release": False,
        "quality_min_joint_valid": 0.95,
        "model1_chemistry_features_used": False,
    }
    artifact = out_dir / "m2_v4_candidate.joblib"
    if artifact.exists():
        raise FileExistsError("Existing final bundle; refuse silent replacement")
    joblib.dump(bundle, artifact)
    if file_sha256(active) != active_sha:
        raise ValueError("Active model registry changed during experiment")
    record = {
        "status": "candidate_not_published_not_activated",
        "winner": winner,
        "samples_total": 706,
        "samples_fit_QA95": int(qa.sum()),
        "label_threshold_ppm_train_only": threshold,
        "n_spectral_features": 170,
        "all_six_competitors_present": True,
        "trained_from": "nested_170_706",
        "source_matrix_sha256": file_sha256(source),
        "source_manifest_sha256": file_sha256(manifest_path),
        "active_registry_sha256_unchanged": active_sha,
        "model_sha256": file_sha256(artifact),
        "calibration_policy": calibration_policy(cfg["optimization"]),
        "test_or_external_holdout_used_for_final_fit": False,
        "release_approved": False,
        "feature_schema_and_new_scene_bands_required_for_inference": True,
        "qa_training_evidence": audit,
    }
    (out_dir / "candidate_metadata.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("M2_V4_CANDIDATE_NOT_RELEASED", json.dumps(record), flush=True)
    return record


if __name__ == "__main__":
    train_final()
