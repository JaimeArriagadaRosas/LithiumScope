"""Audit M2 v4 all six: regional OOF, positive/negative Brier and confusion.

No fitting, no selection of threshold based on held-out labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)

from lithiumscope.core.paths import PROJECT_ROOT
from scripts.scientific.m2_rededge_v4_competition import ARMS, RUNS


FOLDERS = {"fixed": "comparison_706", "nested": "nested_170_706"}


def scores(y, p):
    y = np.asarray(y, int)
    p = np.asarray(p, float)
    if len(np.unique(y)) < 2:
        return {"n": len(y), "positives": int(y.sum()), "auc": None, "ap": None, "brier": float(brier_score_loss(y, p))}
    return {
        "n": len(y),
        "positives": int(y.sum()),
        "auc": float(roc_auc_score(y, p)),
        "ap": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
    }


def matrix(y, p, *, threshold=0.5):
    tn, fp, fn, tp = confusion_matrix(
        np.asarray(y, int),
        np.asarray(p, float) >= threshold,
        labels=[0, 1],
    ).ravel()

    def div(a, b):
        return float(a / b) if b else None

    return {
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "recall": div(tp, tp + fn),
        "specificity": div(tn, tn + fp),
        "precision": div(tp, tp + fp),
        "balanced_accuracy": ((div(tp, tp + fn) + div(tn, tn + fp)) / 2 if tp + fn and tn + fp else None),
    }


def audit(phase="fixed", *, root=PROJECT_ROOT):
    if phase not in FOLDERS:
        raise ValueError("Unsupported scientific phase")
    base = root / "results/scientific_simulations/m2_rededge_v4_20261113"
    folder = base / FOLDERS[phase]
    original = root / "results/model_2/runs" / RUNS[706] / "tables"
    manifest = pd.read_csv(
        root / "data/processed/model_2/training_manifest.csv",
    )
    quality = pd.read_csv(base / "cohort_706/rededge_quality.csv")
    if (
        len(manifest) != 706
        or len(quality) != 706
        or not np.array_equal(manifest.sample_id.astype(str), quality.sample_id.astype(str))
    ):
        raise ValueError("M2 v4 QA and historical sample identities mismatch")
    if phase == "fixed":
        status = json.loads((folder / "experiment_manifest.json").read_text(encoding="utf-8"))
        if status.get("status") != "complete_experimental_development_only":
            raise ValueError("M2 v4 fixed analysis not complete")
    else:
        status = json.loads((folder / "final_status.json").read_text(encoding="utf-8"))
        if status.get("status") != "complete_experimental_no_release":
            raise ValueError("M2 v4 nested analysis not complete")
    ref = pd.read_csv(original / "oof_predictions_random_forest.csv")
    y = ref.y_true.to_numpy(int)
    fold = ref.outer_fold.to_numpy(int)
    if len(y) != 706 or not np.all(np.isin(y, [0, 1])):
        raise ValueError("Invalid labels for M2 v4 scientific audit")
    region = np.where(
        manifest.latitude.between(-20, -10),
        "10-20S",
        np.where(manifest.latitude.between(-30, -20), "20-30S", "other"),
    )
    rows = []
    folds = []
    for algorithm in ARMS:
        latest = np.load(folder / f"oof_{algorithm}.npy", allow_pickle=False)
        earlier = pd.read_csv(original / f"oof_predictions_{algorithm}.csv")
        if (
            len(latest) != 706
            or len(earlier) != 706
            or not np.array_equal(earlier.y_true.to_numpy(int), y)
            or not np.isfinite(latest).all()
        ):
            raise ValueError("M2 v4 missing or unaligned six-family OOF")
        for arm, pred in (
            ("v3_126", earlier.prospectivity_score.to_numpy(float)),
            ("v4_170", latest),
        ):
            for subset, indices in [
                ("all", np.arange(706)),
                ("10-20S", np.flatnonzero(region == "10-20S")),
                ("20-30S", np.flatnonzero(region == "20-30S")),
                ("QA95_rededge", np.flatnonzero(quality.rededge_joint_valid_fraction >= 0.95)),
                ("below_QA95_rededge", np.flatnonzero(quality.rededge_joint_valid_fraction < 0.95)),
            ]:
                if not len(indices):
                    continue
                labels = y[indices]
                p = pred[indices]
                b_pos = float(np.mean((p[labels == 1] - 1) ** 2)) if np.any(labels == 1) else None
                b_neg = float(np.mean(p[labels == 0] ** 2)) if np.any(labels == 0) else None
                rows.append(
                    {
                        "algorithm": algorithm,
                        "representation": arm,
                        "subset": subset,
                        **scores(labels, p),
                        **matrix(labels, p),
                        "brier_positive": b_pos,
                        "brier_negative": b_neg,
                    }
                )
            for k in range(1, 6):
                indices = np.flatnonzero(fold == k)
                folds.append(
                    {
                        "algorithm": algorithm,
                        "representation": arm,
                        "fold": k,
                        **scores(y[indices], pred[indices]),
                        **matrix(y[indices], pred[indices]),
                    }
                )
    folder.joinpath("scientific_audit_matrices_and_regions.csv").write_text(
        pd.DataFrame(rows).to_csv(index=False),
        encoding="utf-8",
    )
    folder.joinpath("scientific_audit_by_fold.csv").write_text(
        pd.DataFrame(folds).to_csv(index=False),
        encoding="utf-8",
    )
    overview = pd.DataFrame(rows).query("subset=='all'")
    pivot = overview.pivot(
        index="algorithm", columns="representation", values=["auc", "ap", "brier", "recall", "FP", "FN"]
    )
    pivot.columns = [f"{a}_{b}" for a, b in pivot.columns]
    pivot["delta_auc"] = pivot["auc_v4_170"] - pivot["auc_v3_126"]
    pivot["delta_ap"] = pivot["ap_v4_170"] - pivot["ap_v3_126"]
    pivot["delta_brier"] = pivot["brier_v4_170"] - pivot["brier_v3_126"]
    pivot.to_csv(folder / "scientific_audit_comparison.csv")
    print("M2_V4_SCIENTIFIC_AUDIT", phase, pivot.round(4).to_string(), flush=True)
    return pivot


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=FOLDERS, default="fixed")
    args = p.parse_args()
    audit(args.phase)


if __name__ == "__main__":
    main()
