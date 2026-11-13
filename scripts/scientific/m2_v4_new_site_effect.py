"""M2 data ablation: all six algorithms, train-only new sites, same 706 TEST.

Requires real 170-D historic matrix and independently downloaded
provisional GEOROC patches. No release promotion and no copied test data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from lithiumscope.core.device import detect_device
from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.paths import PROJECT_ROOT
from lithiumscope.model_2.methodology.rededge_acquisition import cached_rededge_sample
from lithiumscope.model_2.methodology.rededge_training import (
    combine_quality_masks,
    fit_frozen_model,
)
from lithiumscope.model_2.steps.step_07_build_training_set import build_training_set
from lithiumscope.model_2.training.quality import training_quality_mask
from scripts.scientific.m2_rededge_v4_competition import ARMS, _metrics, _splits
from scripts.scientific.m2_rededge_v4_report import write_release_comparison

ROOT = PROJECT_ROOT
WORK = ROOT / "results/scientific_simulations/m2_rededge_v4_20261113"


def distance_from_test_km(new: pd.DataFrame, reference: pd.DataFrame, indices):
    test = reference.iloc[indices]
    tree = BallTree(
        np.deg2rad(test[["latitude", "longitude"]].to_numpy(float)),
        metric="haversine",
    )
    d = (
        tree.query(
            np.deg2rad(new[["latitude", "longitude"]].to_numpy(float)),
            k=1,
        )[0][:, 0]
        * 6371.0088
    )
    return d


def build_new_features(accepted: pd.DataFrame, *, root: Path = ROOT):
    """Exactly the same historical 126-D extractor + 44 actual BOA features."""
    cfg_bands = ("red", "green", "blue", "nir", "swir16", "swir22")
    old = build_training_set(
        accepted,
        lithium_column="Li_icpms",
        spatial_group_column="spatial_group",
        band_names=cfg_bands,
        normalize_per_band=False,
    )
    features = old.drop(columns=["Li_icpms", "spatial_group"])
    if len(features.columns) != 126:
        raise ValueError("New-site legacy spectral block drifted")
    extras = []
    for item in accepted.itertuples(index=False):
        rec, _ = cached_rededge_sample(
            item,
            output=WORK / "new_sites/rededge_4_band_patches",
            min_joint_valid=0.95,
        )
        extras.append(rec)
    added = pd.concat(
        [
            features.reset_index(drop=True),
            pd.DataFrame(extras).reset_index(drop=True),
        ],
        axis=1,
    )
    if added.shape != (len(accepted), 170):
        raise ValueError("New-site v4 matrix shape mismatch")
    if not np.isfinite(added.to_numpy(float)).all():
        raise ValueError("Nonfinite extra site features")
    return added


def run(*, minimum_new: int = 10, buffer_km: float = 5.0, root: Path = ROOT):
    if minimum_new < 1 or buffer_km < 0:
        raise ValueError("Invalid minimum new sites or spatial embargo")
    work = root / "results/scientific_simulations/m2_rededge_v4_20261113"
    accepted_path = work / "new_sites/source_spectra_accepted.csv"
    old_matrix = work / "cohort_706/training_v4_170.csv"
    old_status = work / "cohort_706/build_status.json"
    old_oof = work / "comparison_706"
    if not accepted_path.is_file():
        raise FileNotFoundError("No verified new-site imaging data available")
    if not old_matrix.is_file() or not old_status.is_file():
        raise FileNotFoundError("Historical complete v4 baseline must exist")
    stat = json.loads(old_status.read_text(encoding="utf-8"))
    if stat["status"] != "complete" or stat["matrix_sha256"] != file_sha256(old_matrix):
        raise ValueError("Historical v4 data matrix is incomplete or changed")
    accepted = pd.read_csv(accepted_path)
    if (
        len(accepted) < minimum_new
        or accepted.sample_id.duplicated().any()
        or accepted.spatial_group.duplicated().any()
        or not accepted.scientific_proxy_accepted.all()
        or not accepted.source_label_status.eq("PROVISIONAL_NO_FIELD_VALIDATION").all()
    ):
        raise ValueError("Insufficient or unverified extra-site source cohort")
    historic = pd.read_csv(
        root / "data/processed/model_2/training_manifest.csv",
    )
    if set(accepted.spatial_group) & set(historic.spatial_group):
        raise ValueError("New-site groups overlap historical 706")
    original = pd.read_csv(old_matrix).drop(columns=["Li_icpms", "spatial_group"])
    if original.shape != (706, 170):
        raise ValueError("Historical v4 matrix invalid")
    extra = build_new_features(accepted, root=root)
    q, qa = training_quality_mask(
        root / "data/processed/model_2/training_manifest.csv",
        inventory_path=root
        / ("results/scientific_simulations/sentinel_methodology_20261112/inventory/local_patch_inventory_783.csv"),
        minimum_joint_valid=0.95,
    )
    q, quality_audit = combine_quality_masks(
        q,
        work / "cohort_706/rededge_quality.csv",
        historic,
    )
    folds = _splits(706, historic)
    previous = root / "results/model_2/runs/training_20261113_133905_m0300"
    out = work / "data_ablation_170_new_sites"
    out.mkdir(parents=True, exist_ok=True)
    active = root / "models/active_models.json"
    contract = {
        "base_matrix_sha256": file_sha256(old_matrix),
        "new_site_source_sha256": file_sha256(accepted_path),
        "active_release_registry_sha256": file_sha256(active),
        "all_algorithms": list(ARMS),
        "min_new": minimum_new,
        "buffer_km": buffer_km,
        "new_georoc_labels_are_provisional": True,
        "only_original_706_outer_tests": True,
    }
    previous_contract = out / "experiment_contract.json"
    if previous_contract.exists():
        if json.loads(previous_contract.read_text("utf8")) != contract:
            raise ValueError("New-site ablation contract drifted")
    else:
        previous_contract.write_text(
            json.dumps(contract, indent=2),
            encoding="utf8",
        )
    rows = []
    for alg in ARMS:
        scores = np.full(706, np.nan, float)
        original_model = np.load(
            old_oof / f"oof_{alg}.npy",
            allow_pickle=False,
        )
        new_counts = []
        for k, fold in enumerate(folds, 1):
            near = distance_from_test_km(accepted, historic, fold.test_idx)
            usable = np.flatnonzero(near >= buffer_km)
            add_x = extra.iloc[usable]
            historical_train = fold.train_idx[q[fold.train_idx]]
            fit_x = pd.concat(
                [
                    original.iloc[historical_train],
                    add_x,
                ],
                ignore_index=True,
            )
            labels = np.concatenate(
                [
                    (historic.Li_icpms.iloc[historical_train].to_numpy(float) >= fold.label_threshold_ppm).astype(int),
                    (accepted.Li_icpms.iloc[usable].to_numpy(float) >= fold.label_threshold_ppm).astype(int),
                ]
            )
            group = pd.concat(
                [
                    historic.spatial_group.iloc[historical_train],
                    accepted.spatial_group.iloc[usable],
                ],
                ignore_index=True,
            )
            if set(group) & set(historic.spatial_group.iloc[fold.test_idx]):
                raise ValueError("Spatial group leakage")
            if len(np.unique(labels)) != 2:
                raise ValueError("Extra-site dataset missing train class")
            saved = json.loads((previous / "checkpoints" / alg / f"fold_{k:02d}.json").read_text("utf8"))
            if not np.array_equal(saved["test_indices"], fold.test_idx):
                raise ValueError("Frozen fold index mismatch")
            model = fit_frozen_model(
                alg,
                saved["params"],
                fit_x,
                labels,
                group,
                detect_device(prefer_gpu=False),
                42 + k,
            )
            scores[fold.test_idx] = model.predict_proba(
                original.iloc[fold.test_idx],
            )[:, 1]
            new_counts.append(len(usable))
            print("NEW_SITE_EFFECT", alg, k, len(usable), flush=True)
        if not np.isfinite(scores).all():
            raise ValueError("Outer prediction incomplete")
        np.save(out / f"oof_{alg}.npy", scores, allow_pickle=False)
        y = np.full(706, -1, int)
        for fold in folds:
            y[fold.test_idx] = fold.y_test
        a = _metrics(y, scores)
        b = _metrics(y, original_model)
        rows.append(
            {
                "algorithm": alg,
                "added_train_sites_min": min(new_counts),
                "added_train_sites_max": max(new_counts),
                "new_site_auc": a["roc_auc"],
                "no_new_site_auc": b["roc_auc"],
                "delta_auc": a["roc_auc"] - b["roc_auc"],
                "new_site_ap": a["average_precision"],
                "no_new_site_ap": b["average_precision"],
                "delta_ap": a["average_precision"] - b["average_precision"],
                "new_site_brier": a["brier"],
                "no_new_site_brier": b["brier"],
                "delta_brier": a["brier"] - b["brier"],
            }
        )
    pd.DataFrame(rows).to_csv(
        out / "all_six_new_site_effect.csv",
        index=False,
    )
    comparison = write_release_comparison(out, previous, ARMS, historic)
    if file_sha256(active) != contract["active_release_registry_sha256"]:
        raise ValueError("Active release changed unexpectedly")
    (out / "final_status.json").write_text(
        json.dumps(
            {
                "status": "completed_exploratory_not_a_release",
                "new_sites_accepted": len(accepted),
                "total_source_provisional": True,
                "holdout_independence_not_certified": True,
                "reference_protocol": comparison,
                "original_706_outer_tests_intact": True,
                "quality": quality_audit,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf8",
    )
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--min-new", type=int, default=10)
    p.add_argument("--buffer-km", type=float, default=5)
    args = p.parse_args()
    run(minimum_new=args.min_new, buffer_km=args.buffer_km)


if __name__ == "__main__":
    main()
