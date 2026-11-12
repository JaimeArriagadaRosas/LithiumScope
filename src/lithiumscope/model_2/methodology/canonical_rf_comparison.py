"""Fair M2 spatial comparison: legacy RF versus regularized_v2 RF.

Development CV only; NEVER installs, promotes, replaces or evaluates on the
ten-case demonstration. Outer test rows retain QA-failed observations.
Each arm gets an equal Optuna budget within each outer training split.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, brier_score_loss,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import GroupKFold

from lithiumscope.core.paths import PROJECT_ROOT
from lithiumscope.model_2.evaluation.canonical_protocol import load_protocol, sha256
from lithiumscope.model_2.evaluation.paired_uncertainty import bootstrap_paired_groups
from lithiumscope.model_2.methodology.regularization_study import (
    _apply_calibrator, _calibrator,
)
from lithiumscope.model_2.training import random_forest
from lithiumscope.model_2.training.factory import get_algorithm
from lithiumscope.model_2.training.optimizer import optimize_classifier_with_optuna
from lithiumscope.model_2.training.search_policy import parameter_space_for
from lithiumscope.model_2.evaluation.thresholds import select_operating_threshold

PROFILES = ("legacy", "regularized_v2")


def _metrics(y, score, cutoff: float) -> dict:
    labels = np.asarray(y, dtype=int)
    values = np.asarray(score, dtype=float)
    if (not np.isfinite(values).all()
            or len(np.unique(labels)) != 2):
        raise ValueError("Invalid outer evaluation metrics")
    predicted = values >= cutoff
    return {
        "roc_auc": float(roc_auc_score(labels, values)),
        "average_precision": float(average_precision_score(labels, values)),
        "brier": float(brier_score_loss(labels, values)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "threshold": float(cutoff),
        "predicted_positives": int(np.sum(predicted)),
    }


def _factory(seed: int, params: dict):
    # Keep CPU resources limited and comparable between search arms.
    return random_forest.create_model(random_seed=seed, n_jobs=2, **params)


def _inner_oof(x, y, splits, params: dict, seed: int) -> np.ndarray:
    scores = np.full(len(y), np.nan)
    for number, (train, validation) in enumerate(splits):
        if len(np.unique(y[train])) != 2:
            raise ValueError("Inner training fold is single-class")
        model = _factory(seed + number, params)
        model.fit(x.iloc[train], y[train])
        scores[validation] = model.predict_proba(x.iloc[validation])[:, 1]
    if not np.isfinite(scores).all():
        raise ValueError("Inner OOF predictions are incomplete")
    return scores


def _execute_arm(
    *, manifest, features, fold, eligible, fold_number, profile,
    budget, output: Path, protocol_sha: str,
) -> dict:
    folder = output / f"fold_{fold_number:02d}" / profile
    folder.mkdir(parents=True, exist_ok=True)
    done = folder / "completed.json"
    if done.exists():
        previous = json.loads(done.read_text(encoding="utf-8"))
        if previous["protocol_sha256"] != protocol_sha:
            raise ValueError("Refusing to reuse a different source protocol")
        if previous["trials_budget"] != budget:
            raise ValueError("Trials budget changed; do not mix prior outputs")
        return previous
    train = fold.train_idx[eligible[fold.train_idx]]
    test = fold.test_idx
    groups = manifest.spatial_group.astype(str)
    if set(groups.iloc[train]) & set(groups.iloc[test]):
        raise ValueError("Outer geographic groups overlap")
    x_train = features.iloc[train].reset_index(drop=True)
    x_test = features.iloc[test]
    threshold_li = float(fold.label_threshold_ppm)
    y_train = (manifest.Li_icpms.iloc[train].to_numpy(float) >= threshold_li).astype(int)
    y_test = np.asarray(fold.y_test, dtype=int)
    local_groups = groups.iloc[train].reset_index(drop=True)
    splits = list(GroupKFold(3).split(x_train, y_train, local_groups))
    for tr, va in splits:
        if set(local_groups.iloc[tr]) & set(local_groups.iloc[va]):
            raise ValueError("Inner geographic groups overlap")
        if len(np.unique(y_train[tr])) != 2:
            raise ValueError("Inner train fold has only one class")
    spec = get_algorithm("random_forest")
    search_space = parameter_space_for(
        "random_forest", spec, {"random_forest_search_profile": profile}
    )
    seed = 42 + fold_number * 100
    best = optimize_classifier_with_optuna(
        lambda params: _factory(seed, params),
        search_space, x_train, y_train, splits,
        n_trials=budget, random_seed=seed,
        storage_path=folder / "optuna_spatial_v1.db",
        study_name=f"rf_{profile}_fold_{fold_number}_{protocol_sha[:12]}",
        pruning_enabled=False,
    )
    oof = _inner_oof(x_train, y_train, splits, best, seed)
    calibrator = _calibrator(oof, y_train)
    model = _factory(seed, best)
    model.fit(x_train, y_train)
    scores_raw = model.predict_proba(x_test)[:, 1]
    apparent = model.predict_proba(x_train)[:, 1]
    if not np.isfinite(scores_raw).all():
        raise ValueError("Nonfinite outer scores")
    methods = {
        "raw": (oof, scores_raw),
        "inner_sigmoid": (
            _apply_calibrator(calibrator, oof),
            _apply_calibrator(calibrator, scores_raw),
        ),
    }
    evaluations = {}
    for method, (inside, outside) in methods.items():
        chosen = select_operating_threshold(
            y_train, inside, objective="balanced_accuracy"
        )
        metrics = _metrics(y_test, outside, float(chosen.threshold))
        metrics["apparent_train_auc"] = float(
            roc_auc_score(y_train, apparent)
        )
        metrics["apparent_outer_gap"] = (
            metrics["apparent_train_auc"] - metrics["roc_auc"]
        )
        evaluations[method] = {
            "metrics": metrics,
            "scores": outside.tolist(),
            "selected_using": "inner_grouped_oof_only",
        }
    result = {
        "protocol_sha256": protocol_sha,
        "profile": profile,
        "fold": fold_number,
        "trials_budget": budget,
        "train_eligible": int(len(train)),
        "test_unfiltered": int(len(test)),
        "target_li_ppm_from_outer_train": threshold_li,
        "best_params_inner_optuna": best,
        "test_indices": [int(i) for i in test],
        "test_labels": y_test.tolist(),
        "evaluations": evaluations,
        "model_activated": False,
    }
    staged = folder / "completed.tmp"
    staged.write_text(json.dumps(result, indent=2), encoding="utf-8")
    staged.replace(done)
    return result


def run(*, root: Path, output: Path, trials: int = 16) -> dict:
    if not 1 <= trials <= 100:
        raise ValueError("Choose 1 to 100 trials for each arm and fold")
    manifest, x, quality, _, folds, protocol = load_protocol(root)
    output.mkdir(parents=True, exist_ok=True)
    contract = output / "frozen_protocol.json"
    if contract.exists():
        before = json.loads(contract.read_text(encoding="utf-8"))
        if before["protocol_sha256"] != protocol["protocol_sha256"]:
            raise ValueError("Source drift since the prior development experiment")
    else:
        contract.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    eligible = quality.train_quality_pass_95.to_numpy(dtype=bool)
    runs = []
    for fold_number, fold in enumerate(folds, 1):
        for profile in PROFILES:
            runs.append(_execute_arm(
                manifest=manifest, features=x, fold=fold, eligible=eligible,
                fold_number=fold_number, profile=profile,
                budget=trials, output=output,
                protocol_sha=protocol["protocol_sha256"],
            ))
            print(
                f"FOLD {fold_number}/5 PROFILE {profile} "
                f"ROC-AUC {runs[-1]['evaluations']['raw']['metrics']['roc_auc']:.4f}",
                flush=True,
            )
    if sha256(root / "models/active_models.json") != protocol["input_sha256"]["active_registry"]:
        raise ValueError("The active model registry changed during the experiment")
    rows, metrics = [], []
    for record in runs:
        for method, content in record["evaluations"].items():
            metrics.append({
                "fold": record["fold"], "profile": record["profile"],
                "calibration": method, "train_rows": record["train_eligible"],
                "test_rows": record["test_unfiltered"],
                **content["metrics"],
            })
            for j, idx in enumerate(record["test_indices"]):
                rows.append({
                    "sample_index": idx,
                    "sample_id": str(manifest.sample_id.iloc[idx]),
                    "spatial_group": str(manifest.spatial_group.iloc[idx]),
                    "fold": record["fold"], "profile": record["profile"],
                    "calibration": method,
                    "y_true": record["test_labels"][j],
                    "score": content["scores"][j],
                })
    predictions = pd.DataFrame(rows)
    per_fold = pd.DataFrame(metrics)
    predictions.to_csv(output / "paired_outer_predictions.csv", index=False)
    per_fold.to_csv(output / "fold_metrics.csv", index=False)
    summary = {}
    for method in ("raw", "inner_sigmoid"):
        left = predictions[
            (predictions.profile == "legacy") & (predictions.calibration == method)
        ]
        right = predictions[
            (predictions.profile == "regularized_v2")
            & (predictions.calibration == method)
        ]
        if len(left) != 783 or len(right) != 783:
            raise ValueError("Incomplete paired outer test predictions")
        summary[method] = bootstrap_paired_groups(
            left, right, iterations=400, seed=42
        )
        for profile, frame in (("legacy", left), ("regularized_v2", right)):
            summary[method][profile] = {
                "pooled_auc": float(roc_auc_score(frame.y_true, frame.score)),
                "pooled_ap": float(
                    average_precision_score(frame.y_true, frame.score)
                ),
                "pooled_brier": float(
                    brier_score_loss(frame.y_true, frame.score)
                ),
            }
    report = {
        "status": "development_only_not_eligible_for_release",
        "protocol_sha256": protocol["protocol_sha256"],
        "profiles": list(PROFILES),
        "trials_per_arm_per_fold": trials,
        "outer_test_cases": 783,
        "select_with_external_ten_cases": False,
        "all_legacy_706_folds_preserved": True,
        "not_a_confirmatory_external_holdout": True,
        "comparisons": summary,
    }
    (output / "comparison_summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=16)
    opts = parser.parse_args()
    report = run(root=opts.root, output=opts.output, trials=opts.trials)
    for key, value in report["comparisons"].items():
        print(key, json.dumps(value["metrics"]), flush=True)


if __name__ == "__main__":
    main()
