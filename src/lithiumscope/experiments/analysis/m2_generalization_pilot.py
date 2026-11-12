"""Isolated, deterministic, leakage-aware Model 2 overfit experiment.

Never writes source imagery, candidate model bundles, frozen folds or releases.
Excludes known ten-case demonstration from model/threshold selection.
Usage: python -m lithiumscope.experiments.analysis.m2_generalization_pilot --pilot
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             brier_score_loss, roc_auc_score)
from sklearn.model_selection import GroupKFold

from lithiumscope.core.paths import PROJECT_ROOT
from lithiumscope.model_2.evaluation.thresholds import select_operating_threshold
from lithiumscope.model_2.methodology.independence_gate import (
    audit_external_overlap, pairwise_distances_km, release_gate,
)
from lithiumscope.model_2.methodology.spectral_feature_policy import (
    select_spectral_features,
)

RUN_ID = "training_20261112_123451_m0300"
OUTPUT = "results/scientific_simulations/m2_generalization_20261112"
DATA = "data/processed/model_2/vnext_expanded"
FOLDS = f"results/model_2/runs/{RUN_ID}/manifests/scientific_folds.json"
QA = "results/scientific_simulations/sentinel_methodology_20261112/inventory/local_patch_inventory_783.csv"
DEMO = "results/predictions/demonstration_20261112_164710_m0300/inputs/demonstration_cases.csv"
ARMS = {
    "full126_original_style": ("full_126", False, False, False),
    "direct48_same_complexity": ("direct_band_48", False, False, False),
    "full126_regularized": ("full_126", True, False, False),
    "direct48_regularized": ("direct_band_48", True, False, False),
    "direct48_regularized_buffer5km": ("direct_band_48", True, True, False),
    "direct48_regularized_historical_only": ("direct_band_48", True, False, True),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_contract(root: Path) -> dict:
    paths = {
        "manifest": root / DATA / "training_manifest.csv",
        "features": root / DATA / "spectral_features.csv",
        "quality": root / QA,
        "folds": root / FOLDS,
        "model": root / "models/model_2/trained" / RUN_ID / "model.joblib",
        "external": root / DEMO,
        "active_registry": root / "models/active_models.json",
    }
    for key, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing immutable input: {key}: {path}")
    manifest = pd.read_csv(paths["manifest"])
    features = pd.read_csv(paths["features"])
    quality = pd.read_csv(paths["quality"])
    folds = json.loads(paths["folds"].read_text(encoding="utf-8"))
    bundle = joblib.load(paths["model"])
    if len(manifest) != 783 or len(features) != 783 or len(quality) != 783:
        raise ValueError("This experiment requires exactly 783 frozen rows")
    if not np.array_equal(manifest.sample_id.astype(str), quality.sample_id.astype(str)):
        raise ValueError("Quality inventory not aligned with frozen samples")
    if "sample_id" in features and not np.array_equal(
        manifest.sample_id.astype(str), features.sample_id.astype(str)
    ):
        raise ValueError("Feature cache not aligned with frozen samples")
    if len(folds) != 5:
        raise ValueError("Expected five immutable outer spatial folds")
    mask = quality.sixband_joint_valid_fraction.to_numpy(dtype=float) >= .95
    if mask.sum() != 767:
        raise ValueError("Unexpected quality-eligible training count")
    feature_names = select_spectral_features(bundle["features"], "full_126")
    if not np.isfinite(features[feature_names].to_numpy(dtype=float)).all():
        raise ValueError("Invalid finite feature matrix")
    test_indices = []
    for fold in folds:
        train, test = np.array(fold["train_indices"]), np.array(fold["test_indices"])
        if set(manifest.spatial_group.iloc[train]) & set(manifest.spatial_group.iloc[test]):
            raise ValueError("Spatial train/test group leakage")
        test_indices.extend(test.tolist())
    if sorted(test_indices) != list(range(783)):
        raise ValueError("Outer test cases do not cover unique 783 frozen rows")
    return dict(paths=paths, manifest=manifest, features=features, quality=mask,
                folds=folds, bundle=bundle)


def model_params(bundle: dict, *, regularized: bool) -> dict:
    params = bundle["estimator"].get_params()
    params["n_jobs"] = min(int(params.get("n_jobs") or 4), 4)
    if regularized:
        params.update(max_depth=8, min_samples_leaf=8, n_estimators=240)
    return params


def metrics(y: np.ndarray, scores: np.ndarray, threshold: float = .5) -> dict:
    scores = np.asarray(scores, float)
    y = np.asarray(y, int)
    if len(y) != len(scores) or np.unique(y).size != 2:
        raise ValueError("Metrics require aligned two-class observations")
    return {
        "n": int(len(y)), "positives": int(y.sum()),
        "roc_auc": float(roc_auc_score(y, scores)),
        "average_precision": float(average_precision_score(y, scores)),
        "brier": float(brier_score_loss(y, scores)),
        "balanced_accuracy": float(balanced_accuracy_score(y, scores >= threshold)),
    }


def get_train_indices(c: dict, record: dict, *, buffer5: bool,
                      historical_only: bool) -> np.ndarray:
    train = np.asarray(record["train_indices"], dtype=int)
    test = np.asarray(record["test_indices"], dtype=int)
    train = train[c["quality"][train]]
    if historical_only:
        train = train[train < 706]
    if buffer5:
        distances = pairwise_distances_km(
            c["manifest"].iloc[train], c["manifest"].iloc[test]
        )
        train = train[np.min(distances, axis=0) >= 5.0]
    if len(train) < 150:
        raise ValueError("Not enough quality-and-buffer-eligible training samples")
    return train


def nested_decision(
    x: pd.DataFrame, y: np.ndarray, groups: pd.Series, params: dict,
) -> tuple[LogisticRegression, float, dict]:
    splitter = GroupKFold(n_splits=3)
    scores = np.full(len(y), np.nan, dtype=float)
    for inner_train, inner_test in splitter.split(x, y, groups):
        if len(np.unique(y[inner_train])) < 2 or len(np.unique(y[inner_test])) < 2:
            raise ValueError("Inner calibration group fold lacks both classes")
        inner_model = RandomForestClassifier(**params)
        inner_model.fit(x.iloc[inner_train], y[inner_train])
        scores[inner_test] = inner_model.predict_proba(x.iloc[inner_test])[:, 1]
    if not np.isfinite(scores).all():
        raise ValueError("Incomplete training-only inner OOF scores")
    calibrator = LogisticRegression(C=1.0, max_iter=500)
    calibrator.fit(scores.reshape(-1, 1), y)
    calibrated = calibrator.predict_proba(scores.reshape(-1, 1))[:, 1]
    selected = select_operating_threshold(y, calibrated, objective="balanced_accuracy")
    return calibrator, float(selected.threshold), {
        "inner_training_eligible": int(len(y)),
        "inner_training_groups": int(groups.nunique()),
        "nested_threshold": float(selected.threshold),
        "inner_threshold_BA_optimistic_not_outer_test": selected.objective_value,
    }


def execute_pilot(c: dict, output: Path) -> None:
    records, rows, nested_rows = [], [], []
    names = list(c["bundle"]["features"])
    manifest, features = c["manifest"], c["features"]
    for arm, (policy, regularized, buffer5, historical_only) in ARMS.items():
        columns = select_spectral_features(names, policy)
        params = model_params(c["bundle"], regularized=regularized)
        train_auc_by_fold = []
        for fold_index, record in enumerate(c["folds"], 1):
            train = get_train_indices(
                c, record, buffer5=buffer5, historical_only=historical_only
            )
            test = np.asarray(record["test_indices"], dtype=int)
            lithium_cutoff = float(record["label_threshold_ppm"])
            y_train = (manifest.Li_icpms.iloc[train].to_numpy(float) >= lithium_cutoff).astype(int)
            y_test = (manifest.Li_icpms.iloc[test].to_numpy(float) >= lithium_cutoff).astype(int)
            x_train, x_test = features.iloc[train][columns], features.iloc[test][columns]
            fitted = RandomForestClassifier(**params)
            fitted.fit(x_train, y_train)
            in_sample = fitted.predict_proba(x_train)[:, 1]
            heldout = fitted.predict_proba(x_test)[:, 1]
            train_auc = float(roc_auc_score(y_train, in_sample))
            train_auc_by_fold.append(train_auc)
            frozen_threshold, calibrated_scores = 0.5, None
            if arm == "direct48_regularized":
                cal, frozen_threshold, detail = nested_decision(
                    x_train, y_train, manifest.spatial_group.iloc[train], params,
                )
                calibrated_scores = cal.predict_proba(heldout.reshape(-1, 1))[:, 1]
                nested_rows.append({"arm": arm, "fold": fold_index, **detail})
            row = metrics(y_test, heldout)
            rows.append({
                "arm": arm, "fold": fold_index, "train_n": int(len(train)),
                "test_n": int(len(test)), "train_auc": train_auc,
                "n_features": len(columns), "quality95_train_only": True,
                "buffer5km": buffer5, "georoc_train": int((train >= 706).sum()),
                **row,
            })
            for j, sample_index in enumerate(test):
                records.append({
                    "arm": arm, "fold": fold_index, "sample_index": int(sample_index),
                    "sample_id": str(manifest.sample_id.iloc[sample_index]),
                    "spatial_group": str(manifest.spatial_group.iloc[sample_index]),
                    "li_ppm": float(manifest.Li_icpms.iloc[sample_index]),
                    "y_true_fold": int(y_test[j]), "score": float(heldout[j]),
                    "score_inner_calibrated": (
                        float(calibrated_scores[j]) if calibrated_scores is not None else np.nan
                    ),
                    "frozen_train_only_threshold": frozen_threshold,
                    "test_excluded_from_fit": True,
                })
            print("FOLD",arm,fold_index,"train_auc",round(train_auc,4),
                  "heldout_auc",round(row["roc_auc"],4),flush=True)
        print("ARM_DONE",arm,"MEAN_TRAIN_AUC",round(np.mean(train_auc_by_fold),4),flush=True)
    details = pd.DataFrame(records)
    details.to_csv(output / "paired_fold_predictions.csv", index=False)
    pd.DataFrame(rows).to_csv(output / "per_fold_metrics.csv", index=False)
    pd.DataFrame(nested_rows).to_csv(output / "train_only_nested_thresholds.csv", index=False)
    summary = []
    for arm, group in details.groupby("arm", sort=False):
        group = group.sort_values("sample_index")
        if group.sample_index.tolist() != list(range(783)):
            raise ValueError("Arm did not preserve exactly 783 OOF cases")
        score = group.score.to_numpy(float)
        truth = group.y_true_fold.to_numpy(int)
        item = {"arm": arm, "n_features": len(select_spectral_features(
            names, ARMS[arm][0]
        )), **metrics(truth, score)}
        per_fold = pd.DataFrame(rows).query("arm == @arm")
        item["mean_training_auc"] = float(per_fold.train_auc.mean())
        item["generalization_auc_gap"] = item["mean_training_auc"] - item["roc_auc"]
        if arm == "direct48_regularized":
            calibrated = group.score_inner_calibrated.to_numpy(float)
            cutoffs = group.frozen_train_only_threshold.to_numpy(float)
            item["nested_outer_balanced_accuracy"] = float(
                balanced_accuracy_score(truth, calibrated >= cutoffs)
            )
            item["nested_outer_calibrated_brier"] = float(
                brier_score_loss(truth, calibrated)
            )
        summary.append(item)
    pd.DataFrame(summary).to_csv(output / "pilot_summary.csv", index=False)
    print("PILOT_SUMMARY",pd.DataFrame(summary).to_string(index=False),flush=True)


def run(*, pilot: bool = False, root: Path = PROJECT_ROOT) -> Path:
    c = load_contract(root)
    output = root / OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    inputs = {key: sha256(value) for key, value in c["paths"].items()}
    external = pd.read_csv(c["paths"]["external"])
    overlap, independence = audit_external_overlap(c["manifest"], external)
    overlap.to_csv(output / "external_overlap_audit.csv", index=False)
    (output / "independence_summary.json").write_text(
        json.dumps(independence, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    gate = release_gate(independence)
    (output / "scientific_release_gate.json").write_text(
        json.dumps(gate, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    source = {
        "experiment": "no_release_no_retraining_existing_models",
        "root": str(root), "inputs_sha256": inputs,
        "arms_predeclared": ARMS, "candidate_algorithm": c["bundle"]["algorithm"],
        "controlled_fold_labels": "frozen_outer_training_only_historical_q75",
        "outer_test_cases": 783, "train_qa95_eligible": 767,
        "selected_technique": "direct_sentinel2_band_features_48",
        "demo_used_for_tuning": False, "active_models_changed": False,
        "independent_holdout": False, "release_allowed": False,
        "experiment_is_descriptive_not_confirmatory": True,
    }
    (output / "experiment_protocol.json").write_text(
        json.dumps(source, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("DIAGNOSTIC",independence,"RELEASE_GATE",gate,flush=True)
    if pilot:
        execute_pilot(c, output)
    if sha256(c["paths"]["active_registry"]) != inputs["active_registry"]:
        raise RuntimeError("Active models unexpectedly changed during scientific experiment")
    for name in ("features", "manifest", "quality", "folds", "model"):
        if sha256(c["paths"][name]) != inputs[name]:
            raise RuntimeError(f"Immutable source changed during experiment: {name}")
    print("READ_ONLY_SOURCE_CONTRACT_OK",output,flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true", help="Run controlled five-fold RF arms")
    args = parser.parse_args()
    run(pilot=args.pilot)


if __name__ == "__main__":
    main()
