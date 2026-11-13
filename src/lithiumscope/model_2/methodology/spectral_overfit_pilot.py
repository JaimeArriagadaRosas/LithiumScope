"""M2 spectral-only overfitting ablation on five frozen spatial folds.

No changes to models, chemical M1 features, source patches or interactive menu.
Test imagery is strictly original; auxiliary gain variants exist for TRAIN only.
All results are REUSED DEVELOPMENT FOLDS, not external validation or release.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from lithiumscope.core.paths import PROJECT_ROOT
from lithiumscope.model_2.evaluation.canonical_protocol import load_protocol, sha256
from lithiumscope.model_2.evaluation.paired_uncertainty import bootstrap_paired_groups
from lithiumscope.model_2.evaluation.spatial_distance_sensitivity import distances_km
from lithiumscope.model_2.methodology.spectral_feature_policy import select_spectral_features
from lithiumscope.model_2.methodology.spectral_overfit_inputs import build_augmentation_cache
from lithiumscope.model_2.training.random_forest import create_model

ARMS = {
    "control_regularized_126": (False, "all126", False),
    "gain_augment_train_126": (True, "all126", False),
    "predeclared_direct_48": (False, "direct48", False),
    "train_only_kbest_48": (False, "kbest48", False),
    "gain_plus_kbest_48": (True, "kbest48", False),
    "true_5km_train_embargo": (False, "all126", True),
}
FIXED_RF = {
    "n_estimators": 250,
    "max_depth": 8,
    "min_samples_leaf": 10,
    "max_features": "sqrt",
    "class_weight": "balanced",
    "n_jobs": 2,
}


def embargo_training_indices(
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    coordinates: np.ndarray,
    *,
    km: float = 5.0,
) -> np.ndarray:
    if km < 0:
        raise ValueError("Distance must be nonnegative")
    min_distance = distances_km(coordinates[train_idx], coordinates[test_idx]).min(axis=1)
    return train_idx[min_distance >= km]


def choose_columns(
    x_train: pd.DataFrame, labels: np.ndarray, strategy: str
) -> list[str]:
    if strategy == "all126":
        if x_train.shape[1] != 126:
            raise ValueError("Expected frozen 126-D M2 spectra")
        return list(x_train.columns)
    if strategy == "direct48":
        return select_spectral_features(list(x_train.columns), "direct_band_48")
    if strategy == "kbest48":
        selector = SelectKBest(score_func=f_classif, k=48)
        selector.fit(x_train, labels)
        selected = x_train.columns[selector.get_support()].tolist()
        if len(selected) != 48:
            raise ValueError("Train-only feature filter changed selected count")
        return selected
    raise ValueError(f"Unknown spectral column strategy: {strategy}")


def train_and_score(
    original: pd.DataFrame,
    augmented: pd.DataFrame,
    labels: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    strategy: str,
    augment_train: bool,
    seed: int,
):
    train = original.iloc[train_idx]
    selected = choose_columns(train, labels, strategy)
    fit_x = train[selected]
    fit_y = labels
    weights = None
    if augment_train:
        copied = augmented.iloc[train_idx][selected]
        if not copied.index.equals(fit_x.index):
            raise ValueError("Augmented rows lost parent ordering")
        fit_x = pd.concat([fit_x, copied], ignore_index=True)
        fit_y = np.concatenate([labels, labels])
        weights = np.full(len(fit_y), 0.5, dtype=float)
    classifier = create_model(random_seed=seed, **FIXED_RF)
    classifier.fit(fit_x, fit_y, sample_weight=weights)
    outer = classifier.predict_proba(original.iloc[test_idx][selected])[:, 1]
    apparent = classifier.predict_proba(train[selected])[:, 1]
    if not np.isfinite(outer).all() or not np.isfinite(apparent).all():
        raise ValueError("Nonfinite spectral prediction")
    return outer, apparent, selected


def run(root: Path, output: Path, *, bootstrap: int = 200) -> dict:
    if bootstrap < 40:
        raise ValueError("Require at least forty clustered bootstrap iterations")
    root = root.resolve()
    manifest, features, quality, _, folds, protocol = load_protocol(root)
    output.mkdir(parents=True, exist_ok=True)
    contract = output / "frozen_spectral_study.json"
    planned = {
        "study_version": "train_only_sentinel_dn_gain_and_feature_reduction_v1",
        "canonical_protocol_sha256": protocol["protocol_sha256"],
        "source_hashes": protocol["input_sha256"],
        "arms": {name: list(spec) for name, spec in ARMS.items()},
        "fixed_rf_hyperparameters": FIXED_RF,
        "bootstrap_iterations": bootstrap,
        "no_m1_features": True,
        "no_new_ground_truth": True,
        "no_model_activation": True,
    }
    if contract.exists():
        if json.loads(contract.read_text(encoding="utf-8")) != planned:
            raise ValueError("Frozen M2 study settings or sources differ")
    else:
        contract.write_text(json.dumps(planned, indent=2), encoding="utf-8")
    augmented = build_augmentation_cache(
        manifest, features, output / "train_only_augmentation",
        protocol_sha=protocol["protocol_sha256"],
    )
    eligible = quality.train_quality_pass_95.to_numpy(dtype=bool)
    groups = manifest.spatial_group.astype(str)
    coordinates = manifest[["latitude", "longitude"]].to_numpy(float)
    cases = []
    fold_report = []
    selectors = []
    for k, fold in enumerate(folds, start=1):
        test = np.asarray(fold.test_idx, dtype=int)
        base_train = fold.train_idx[eligible[fold.train_idx]]
        if set(groups.iloc[base_train]) & set(groups.iloc[test]):
            raise ValueError("Outer spatial group leakage")
        y_test = np.asarray(fold.y_test, dtype=int)
        for arm, (aug, policy, embargo) in ARMS.items():
            train = (
                embargo_training_indices(base_train, test, coordinates)
                if embargo else base_train.copy()
            )
            y_train = (
                manifest.Li_icpms.iloc[train].to_numpy(float)
                >= fold.label_threshold_ppm
            ).astype(int)
            if len(set(y_train)) != 2 or len(set(y_test)) != 2:
                raise ValueError(f"Missing class for {arm} fold {k}")
            if set(train) & set(test):
                raise ValueError("Train/test row leakage")
            probabilities, apparent, columns = train_and_score(
                features, augmented, y_train, train, test,
                strategy=policy, augment_train=aug, seed=42 + k,
            )
            test_auc = float(roc_auc_score(y_test, probabilities))
            fit_auc = float(roc_auc_score(y_train, apparent))
            fold_report.append({
                "fold": k, "arm": arm, "train_unique_samples": int(len(train)),
                "fit_rows_with_augmented_copies": int(len(train) * (1 + int(aug))),
                "removed_by_true_buffer": int(len(base_train) - len(train)),
                "test_original_samples": int(len(test)), "features": len(columns),
                "test_roc_auc": test_auc,
                "test_average_precision": float(average_precision_score(y_test, probabilities)),
                "test_brier": float(brier_score_loss(y_test, probabilities)),
                "apparent_train_auc": fit_auc, "apparent_auc_gap": fit_auc - test_auc,
            })
            selectors.append({"fold": k, "arm": arm, "selected_columns": columns})
            for loc, idx in enumerate(test):
                cases.append({
                    "fold": k, "arm": arm, "sample_index": int(idx),
                    "sample_id": str(manifest.sample_id.iloc[idx]),
                    "spatial_group": str(groups.iloc[idx]),
                    "y_true": int(y_test[loc]), "score": float(probabilities[loc]),
                })
        print(f"SPECTRAL_OVERFIT_FOLD_COMPLETED {k}/{len(folds)}", flush=True)
    pred = pd.DataFrame(cases)
    per_fold = pd.DataFrame(fold_report)
    counts = pred.groupby("arm").sample_index.agg(["count", "nunique"])
    if not ((counts["count"] == 783).all() and (counts["nunique"] == 783).all()):
        raise ValueError("Some spectral arms lost or duplicated external test cases")
    if sha256(root / "models/active_models.json") != protocol["input_sha256"]["active_registry"]:
        raise ValueError("Active model registry changed")
    pred.to_csv(output / "paired_outer_predictions.csv", index=False)
    per_fold.to_csv(output / "fold_metrics.csv", index=False)
    (output / "selected_features_by_outer_train.json").write_text(
        json.dumps(selectors, indent=2), encoding="utf-8"
    )
    report = {}
    for arm in ARMS:
        selected = pred[pred.arm == arm]
        y = selected.y_true.to_numpy(int)
        score = selected.score.to_numpy(float)
        report[arm] = {
            "n_test": len(selected),
            "roc_auc_pooled": float(roc_auc_score(y, score)),
            "average_precision_pooled": float(average_precision_score(y, score)),
            "brier_pooled": float(brier_score_loss(y, score)),
            "mean_apparent_train_auc_gap": float(
                per_fold.loc[per_fold.arm == arm, "apparent_auc_gap"].mean()
            ),
        }
    reference = pred[pred.arm == "control_regularized_126"]
    comparisons = {}
    for arm in ARMS:
        if arm == "control_regularized_126":
            continue
        comparisons[arm] = bootstrap_paired_groups(
            reference, pred[pred.arm == arm], iterations=bootstrap, seed=20261112
        )
    final = {
        "state": "development_only_completed_no_release",
        "protocol_sha256": protocol["protocol_sha256"],
        "original_rows_per_arm": 783,
        "all_5_frozen_spatial_folds_held": True,
        "test_images_never_perturbed": True,
        "no_chemical_feature_input": True,
        "augmented_samples_are_not_independent": True,
        "fixed_params_no_outer_test_tuning": True,
        "models_activated": False,
        "arms": report,
        "paired_cluster_bootstrap": comparisons,
        "warning": "Fold reuse is exploratory. No external confirmatory holdout.",
    }
    (output / "study_summary.json").write_text(
        json.dumps(final, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("SPECTRAL_OVERFIT_STUDY_COMPLETE", json.dumps(report), flush=True)
    return final


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=200)
    args = parser.parse_args()
    run(args.root, args.output, bootstrap=args.bootstrap)


if __name__ == "__main__":
    main()
