"""One-shot M2 closeout vs the actually active legacy release.

Uses a fixed, already-executed M2 candidate and the legacy archived OOF;
optional sigmoid is learned strictly in each outer TRAIN group. Does not
retrain a deployed bundle, mutate files, select on demonstration or publish.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from lithiumscope.core.paths import PROJECT_ROOT
from lithiumscope.model_2.evaluation.canonical_protocol import load_protocol, sha256
from lithiumscope.model_2.evaluation.paired_uncertainty import bootstrap_paired_groups
from lithiumscope.model_2.methodology.regularization_study import (
    _apply_calibrator, _calibrator, _inner_scores,
)
from lithiumscope.model_2.methodology.spectral_overfit_pilot import FIXED_RF
from lithiumscope.model_2.training.random_forest import create_model

LEGACY_OOF = ("results/scientific_simulations/sentinel_methodology_20261112/"
              "experiments/all_arms_paired_oof.csv")
NEW_OOF = ("results/scientific_simulations/m2_spectral_overfit_20261112/"
           "paired_outer_predictions.csv")
RELEASE_ID = "training_20260920_195933_m0300"
CANDIDATE_NAME = "control_regularized_126"


def read_paired(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    registry = json.loads((root / "models/active_models.json").read_text("utf-8"))
    active_path = registry["models"]["model_2"]["model_path"].replace("\\", "/")
    if RELEASE_ID not in active_path or "model_1" in active_path:
        raise ValueError(f"Active M2 release changed: {active_path}")
    previous = pd.read_csv(root / LEGACY_OOF)
    previous = previous.loc[previous.arm == "legacy_archived"].copy()
    previous.rename(columns={"test_index": "sample_index"}, inplace=True)
    candidates = pd.read_csv(root / NEW_OOF)
    candidate = candidates.loc[
        (candidates.arm == CANDIDATE_NAME)
        & (candidates.sample_index < 706)
    ].copy()
    fields = ["sample_index", "fold", "sample_id", "spatial_group", "y_true"]
    if len(previous) != 706 or len(candidate) != 706:
        raise ValueError("Expected exactly 706 original release test positions")
    left = previous.sort_values("sample_index").reset_index(drop=True)
    right = candidate.sort_values("sample_index").reset_index(drop=True)
    for field in fields:
        if not np.array_equal(left[field].astype(str), right[field].astype(str)):
            raise ValueError(f"Not the same released test cases: {field}")
    if list(left.sample_index) != list(range(706)):
        raise ValueError("Archive is not complete test holdout")
    if previous.spatial_group.nunique() < 5:
        raise ValueError("Insufficient independent geographic groups")
    return left, right


def train_only_sigmoid(root: Path, candidate: pd.DataFrame) -> pd.DataFrame:
    manifest, features, quality, _, folds, _ = load_protocol(root)
    eligible = quality.train_quality_pass_95.to_numpy(bool)
    groups = manifest.spatial_group.astype(str)
    calibrated = candidate.copy()
    values = {}
    for number, fold in enumerate(folds, start=1):
        train = fold.train_idx[eligible[fold.train_idx]]
        test = fold.test_idx
        if set(groups.iloc[train]) & set(groups.iloc[test]):
            raise ValueError("Train/test geographic group overlap")
        y_train = (
            manifest.Li_icpms.iloc[train].to_numpy(float)
            >= fold.label_threshold_ppm
        ).astype(int)
        estimator = create_model(random_seed=42 + number, **FIXED_RF)
        train_features = features.iloc[train].reset_index(drop=True)
        inner_scores = _inner_scores(
            estimator, train_features, y_train,
            groups.iloc[train].reset_index(drop=True),
            seed=20261112 + number * 3,
        )
        calibrator = _calibrator(inner_scores, y_train)
        sub = candidate.loc[candidate.fold == number]
        values.update(dict(zip(
            sub.sample_index.to_numpy(int),
            _apply_calibrator(calibrator, sub.score.to_numpy(float)),
        )))
        print(f"RELEASE_CLOSEOUT_CALIBRATION {number}/5", flush=True)
    if set(values) != set(candidate.sample_index.to_numpy(int)):
        raise ValueError("Missing sigmoid calibrated cases")
    calibrated["score"] = calibrated.sample_index.map(values).astype(float)
    return calibrated


def metric_summary(frame: pd.DataFrame) -> dict:
    y, p = frame.y_true.to_numpy(int), frame.score.to_numpy(float)
    if not np.isfinite(p).all() or len(np.unique(y)) != 2:
        raise ValueError("Invalid or single-class scored release test")
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "rows": len(y), "positives": int(y.sum()),
    }


def run(root: Path, output: Path, *, repetitions: int = 320) -> dict:
    root = root.resolve()
    previous, candidate = read_paired(root)
    calibrated = train_only_sigmoid(root, candidate)
    output.mkdir(parents=True, exist_ok=True)
    candidate = candidate.assign(arm="corrected_fixed_rf_raw")
    calibrated = calibrated.assign(arm="corrected_fixed_rf_train_sigmoid")
    previous = previous.assign(arm="active_release_archived_oof")
    cases = pd.concat([previous, candidate, calibrated], ignore_index=True)
    cases.to_csv(output / "paired_release_706.csv", index=False)
    estimates = {
        arm: metric_summary(frame)
        for arm, frame in cases.groupby("arm", sort=False)
    }
    deltas = {}
    for name, frame in (("corrected_fixed_rf_raw", candidate),
                        ("corrected_fixed_rf_train_sigmoid", calibrated)):
        paired = bootstrap_paired_groups(
            previous, frame, iterations=repetitions, seed=20261112
        )
        deltas[name] = paired
    manifest = {
        "status": "closeout_exploratory_not_release_certification",
        "active_release": RELEASE_ID,
        "reference_type": "archived_original_frozen_706_OOF_predictions",
        "matched_706_indices_labels_ids_groups_and_folds": True,
        "common_test_set_was_reused_during_model_development": True,
        "scores_obtained_from": {
            "previous": LEGACY_OOF, "new": NEW_OOF,
        },
        "model_1_untouched": True,
        "menu_untouched": True,
        "active_registry_sha256": sha256(root / "models/active_models.json"),
        "metrics": estimates,
        "paired_bootstrap_by_spatial_group": deltas,
        "decision": {
            "improves_raw_auc_and_ap": (
                estimates["corrected_fixed_rf_raw"]["roc_auc"]
                > estimates["active_release_archived_oof"]["roc_auc"]
                and estimates["corrected_fixed_rf_raw"]["average_precision"]
                > estimates["active_release_archived_oof"]["average_precision"]
            ),
            "improves_every_metric_calibrated": all((
                estimates["corrected_fixed_rf_train_sigmoid"][name]
                < estimates["active_release_archived_oof"][name]
                if name == "brier" else
                estimates["corrected_fixed_rf_train_sigmoid"][name]
                > estimates["active_release_archived_oof"][name]
            ) for name in ("roc_auc", "average_precision", "brier")),
            "eligible_to_claim_external_superiority": False,
            "release_activation": False,
        },
    }
    (output / "closure_decision.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("M2_RELEASE_CLOSEOUT", json.dumps({
        "metrics": estimates, "decision": manifest["decision"],
        "intervals": {
            k: {m: v["ci95"] for m, v in p["metrics"].items()}
            for k, p in deltas.items()
        },
    }), flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repetitions", type=int, default=320)
    args = parser.parse_args()
    run(args.root, args.output, repetitions=args.repetitions)


if __name__ == "__main__":
    main()
