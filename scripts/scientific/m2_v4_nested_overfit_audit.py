"""Reproduce outer fold estimators and quantify TRAIN/TEST AUC gaps for M2 v4.

All six algorithms; no Optuna, no changes to folds, labels or releases.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

from lithiumscope.core.device import detect_device
from lithiumscope.core.paths import PROJECT_ROOT
from lithiumscope.model_2.training.cv_runner_fold_utilities import _fit_estimator
from lithiumscope.model_2.training.quality import training_quality_mask
from lithiumscope.model_2.methodology.rededge_training import combine_quality_masks

from scripts.scientific.m2_rededge_v4_competition import ARMS, _splits


def audit(*, root: Path = PROJECT_ROOT) -> pd.DataFrame:
    base = root / "results/scientific_simulations/m2_rededge_v4_20261113"
    trained = base / "nested_170_706"
    status_path = trained / "final_status.json"
    if not status_path.is_file():
        raise FileNotFoundError("Nested competition has not finished")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("status") != "complete_experimental_no_release":
        raise ValueError("Cannot analyze partial nested competition")
    frame = pd.read_csv(base / "cohort_706/training_v4_170.csv")
    manifest = pd.read_csv(root / "data/processed/model_2/training_manifest.csv")
    if len(frame) != 706 or len(manifest) != 706 or len(frame.columns) != 172:
        raise ValueError("Historical data drifted before overfit audit")
    x = frame.drop(columns=["Li_icpms", "spatial_group"])
    original_quality, _ = training_quality_mask(
        root / "data/processed/model_2/training_manifest.csv",
        inventory_path=root
        / ("results/scientific_simulations/sentinel_methodology_20261112/inventory/local_patch_inventory_783.csv"),
        minimum_joint_valid=0.95,
    )
    eligible, _ = combine_quality_masks(
        original_quality,
        base / "cohort_706/rededge_quality.csv",
        manifest,
    )
    folds = _splits(706, manifest)
    device = detect_device(prefer_gpu=False)
    rows = []
    for algorithm in ARMS:
        oof = np.load(trained / f"oof_{algorithm}.npy", allow_pickle=False)
        if len(oof) != 706:
            raise ValueError("Missing nested OOF model " + algorithm)
        for k, fold in enumerate(folds, 1):
            checkpoint = trained / "checkpoints" / algorithm / f"fold_{k:02d}.json"
            state = json.loads(checkpoint.read_text(encoding="utf-8"))
            if not np.array_equal(state["test_indices"], fold.test_idx):
                raise ValueError("Refit fold TEST drifted")
            fit = fold.train_idx[eligible[fold.train_idx]]
            train_y = (manifest.Li_icpms.iloc[fit].to_numpy(float) >= fold.label_threshold_ppm).astype(int)
            if len(np.unique(train_y)) < 2:
                raise ValueError("TRAIN with insufficient class coverage")
            model = _fit_estimator(
                algorithm,
                device,
                42,
                state["params"],
                x.iloc[fit],
                pd.Series(train_y),
                manifest.spatial_group.iloc[fit],
                3,
                strict_spatial=True,
            )
            p_tr = np.asarray(model.predict_proba(x.iloc[fit])[:, 1], float)
            p_te = np.asarray(model.predict_proba(x.iloc[fold.test_idx])[:, 1], float)
            numerical = float(np.max(np.abs(p_te - oof[fold.test_idx])))
            y_test = np.asarray(fold.y_test, int)
            rows.append(
                {
                    "algorithm": algorithm,
                    "fold": k,
                    "train_n": len(fit),
                    "test_n": len(fold.test_idx),
                    "train_auc": float(roc_auc_score(train_y, p_tr)),
                    "test_auc": float(roc_auc_score(y_test, p_te)),
                    "train_ap": float(average_precision_score(train_y, p_tr)),
                    "test_ap": float(average_precision_score(y_test, p_te)),
                    "train_brier": float(brier_score_loss(train_y, p_tr)),
                    "test_brier": float(brier_score_loss(y_test, p_te)),
                    "maximum_refit_vs_saved_oof_difference": numerical,
                    "outer_test_never_used_for_fit": True,
                }
            )
            print(
                "M2_V4_OVERFIT",
                algorithm,
                k,
                "AUC_TRAIN",
                round(rows[-1]["train_auc"], 4),
                "AUC_TEST",
                round(rows[-1]["test_auc"], 4),
                "OOF_DIFF",
                numerical,
                flush=True,
            )
    data = pd.DataFrame(rows)
    data["auc_generalization_gap"] = data.train_auc - data.test_auc
    data["ap_generalization_gap"] = data.train_ap - data.test_ap
    data["brier_generalization_gap"] = data.test_brier - data.train_brier
    data.to_csv(trained / "overfitting_diagnostic_all_six_folds.csv", index=False)
    summary = (
        data.groupby("algorithm")
        .agg(
            train_auc_mean=("train_auc", "mean"),
            test_auc_mean=("test_auc", "mean"),
            auc_gap_mean=("auc_generalization_gap", "mean"),
            train_ap_mean=("train_ap", "mean"),
            test_ap_mean=("test_ap", "mean"),
            ap_gap_mean=("ap_generalization_gap", "mean"),
            brier_train_mean=("train_brier", "mean"),
            brier_test_mean=("test_brier", "mean"),
            max_refit_diff=("maximum_refit_vs_saved_oof_difference", "max"),
        )
        .reset_index()
    )
    summary.to_csv(trained / "overfitting_diagnostic_summary.csv", index=False)
    print("M2_V4_NESTED_OVERFIT_COMPLETE", summary.to_string(index=False), flush=True)
    return summary


if __name__ == "__main__":
    audit()
