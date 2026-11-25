"""Audit proximity across frozen spatial folds, without changing their tests.

Supplementary sensitivity only: excluding nearby TEST cases for reporting
does not substitute for a future genuinely buffered re-training experiment.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from lithiumscope.core.paths import PROJECT_ROOT
from lithiumscope.model_2.evaluation.canonical_protocol import load_protocol


def distances_km(test_coordinates, train_coordinates) -> np.ndarray:
    """Geodesic distances for [lat, lon] arrays in degrees."""
    test = np.radians(np.asarray(test_coordinates, dtype=float))
    train = np.radians(np.asarray(train_coordinates, dtype=float))
    if test.ndim != 2 or train.ndim != 2:
        raise ValueError("Expect two-dimensional coordinates")
    if test.shape[1] != 2 or train.shape[1] != 2:
        raise ValueError("Lat/lon pairs required")
    if not np.isfinite(test).all() or not np.isfinite(train).all():
        raise ValueError("Nonfinite coordinates")
    dlat = test[:, None, 0] - train[None, :, 0]
    dlon = test[:, None, 1] - train[None, :, 1]
    hav = (
        np.sin(dlat / 2) ** 2
        + np.cos(test[:, None, 0]) * np.cos(train[None, :, 0])
        * np.sin(dlon / 2) ** 2
    )
    return 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(hav, 0.0, 1.0)))


def audit_nearest_training(root: Path) -> pd.DataFrame:
    manifest, _, quality, _, folds, _ = load_protocol(root)
    mask = quality.train_quality_pass_95.to_numpy(dtype=bool)
    coords = manifest[["latitude", "longitude"]].to_numpy(float)
    rows = []
    for number, fold in enumerate(folds, 1):
        training = fold.train_idx[mask[fold.train_idx]]
        test = fold.test_idx
        distance = distances_km(coords[test], coords[training])
        nearest = distance.argmin(axis=1)
        for index, training_index in enumerate(nearest):
            rows.append({
                "sample_index": int(test[index]),
                "sample_id": str(manifest.sample_id.iloc[test[index]]),
                "fold": number,
                "nearest_train_sample_id": str(
                    manifest.sample_id.iloc[training[training_index]]
                ),
                "nearest_train_km": float(distance[index, training_index]),
                "near_1km": bool(distance[index, training_index] < 1),
                "near_5km": bool(distance[index, training_index] < 5),
            })
    result = pd.DataFrame(rows).sort_values("sample_index")
    if len(result) != 783 or result.sample_index.nunique() != 783:
        raise ValueError("Proximity audit must cover all test cases once")
    return result


def evaluate_distance_sensitivity(
    prediction_csv: Path, distance: pd.DataFrame
) -> dict:
    predictions = pd.read_csv(prediction_csv)
    data = predictions.merge(
        distance[["sample_index", "nearest_train_km"]],
        on="sample_index", validate="many_to_one"
    )
    if len(data) != 783 * 4:
        raise ValueError("Paired study must have 783 rows for each of four arms")
    result = {}
    for name, selected in {
        "all_783": data,
        "at_least_5km": data[data.nearest_train_km >= 5],
        "under_5km": data[data.nearest_train_km < 5],
    }.items():
        grouped = {}
        for (profile, calibration), frame in selected.groupby(
            ["profile", "calibration"]
        ):
            y = frame.y_true.to_numpy(int)
            score = frame.score.to_numpy(float)
            if len(np.unique(y)) != 2:
                grouped[f"{profile}/{calibration}"] = {
                    "n": len(y), "scoring_available": False
                }
                continue
            grouped[f"{profile}/{calibration}"] = {
                "n": len(y),
                "positives": int(y.sum()),
                "roc_auc": float(roc_auc_score(y, score)),
                "average_precision": float(average_precision_score(y, score)),
                "brier": float(brier_score_loss(y, score)),
            }
        result[name] = grouped
    return result


def run(*, root: Path, comparison: Path) -> dict:
    if not (comparison / "comparison_summary.json").is_file():
        raise ValueError("Complete paired comparison before sensitivity analysis")
    nearest = audit_nearest_training(root)
    nearest.to_csv(comparison / "outer_case_train_proximity.csv", index=False)
    result = {
        "state": "exploratory_distance_subset_not_a_buffered_cv",
        "n_outer_cases": len(nearest),
        "near_1km": int(nearest.near_1km.sum()),
        "near_5km": int(nearest.near_5km.sum()),
        "far_5km": int((~nearest.near_5km).sum()),
        "per_fold_near_5km": (
            nearest.groupby("fold").near_5km.sum().astype(int).to_dict()
        ),
        "metrics": evaluate_distance_sensitivity(
            comparison / "paired_outer_predictions.csv", nearest
        ),
        "limitations": (
            "This does not remove nearby TRAIN samples before fitting; "
            "it scores a less-near subset only. For strict spatial buffering, "
            "retrain each fold with an embargo, preserving all TEST cases."
        ),
    }
    (comparison / "distance_sensitivity.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--comparison", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(root=args.root, comparison=args.comparison), indent=2))


if __name__ == "__main__":
    main()
