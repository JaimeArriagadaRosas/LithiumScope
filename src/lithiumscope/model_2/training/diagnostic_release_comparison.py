"""Matched historical 706-case comparison of ALL six M2 arms to release protocol.

Reference is refitted historical CatBoost release HYPERPARAMETERS, never
the fitted released model evaluated in-sample. No implicit release approval.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from lithiumscope.model_2.training.diagnostic_metrics import (
    binary_diagnostic, spatial_bootstrap_difference,
)

REFERENCE = "original_release_protocol_retrained"


def compare_to_release_protocol(tables: Path, output: Path, *, repeats=160) -> dict:
    source = tables / "old_release_protocol_paired_706.csv"
    if not source.is_file():
        return {"status": "missing_historical_reference", "release_pass": False}
    frame = pd.read_csv(source)
    keys = ["test_index", "fold", "sample_id", "spatial_group", "y_true"]
    before = frame.loc[frame.arm == REFERENCE].sort_values(
        "test_index"
    ).reset_index(drop=True)
    if len(before) != 706 or list(before.test_index) != list(range(706)):
        raise ValueError("Missing original 706 historical reference cases")
    rows = []
    baseline = binary_diagnostic(before.y_true, before.score)
    for arm, candidate in frame.groupby("arm"):
        if arm == REFERENCE:
            continue
        current = candidate.sort_values("test_index").reset_index(drop=True)
        if len(current) != 706:
            raise ValueError(f"Missing historical candidate cases for {arm}")
        for field in keys:
            if not np.array_equal(
                before[field].astype(str), current[field].astype(str)
            ):
                raise ValueError(f"Historical mismatch for {arm} field {field}")
        prediction = binary_diagnostic(current.y_true, current.score)
        confidence = spatial_bootstrap_difference(
            before.y_true.to_numpy(int),
            before.score.to_numpy(float),
            current.score.to_numpy(float),
            before.spatial_group.to_numpy(str),
            n=repeats, seed=20261112,
        )
        row = {
            "algorithm": arm, "reference": REFERENCE,
            "reference_auc": baseline["roc_auc"],
            "candidate_auc": prediction["roc_auc"],
            "reference_ap": baseline["average_precision"],
            "candidate_ap": prediction["average_precision"],
            "reference_brier": baseline["brier"],
            "candidate_brier": prediction["brier"],
            "reference_VP": baseline["VP"],
            "reference_VN": baseline["VN"],
            "reference_FP": baseline["FP"],
            "reference_FN": baseline["FN"],
            "candidate_VP": prediction["VP"],
            "candidate_VN": prediction["VN"],
            "candidate_FP": prediction["FP"],
            "candidate_FN": prediction["FN"],
            "auc_gain": confidence["roc_auc"]["delta"],
            "ap_gain": confidence["average_precision"]["delta"],
            "brier_change": confidence["brier"]["delta"],
            "auc_ci95": json.dumps(confidence["roc_auc"]["ci95"]),
            "ap_ci95": json.dumps(confidence["average_precision"]["ci95"]),
            "brier_ci95": json.dumps(confidence["brier"]["ci95"]),
            "internal_auc_ap_gain": (
                confidence["roc_auc"]["delta"] > 0 and
                confidence["average_precision"]["delta"] > 0
            ),
            "internal_calibration_not_worse": (
                confidence["brier"]["delta"] <= 0
            ),
            "external_release_approved": False,
        }
        rows.append(row)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(
        output / "historical_release_comparison_six_algorithms.csv",
        index=False,
    )
    return {
        "status": "matched_development_comparison",
        "historical_test_cases": 706,
        "algorithm_count": len(rows),
        "reference": REFERENCE,
        "original_fitted_release_never_scored_in_sample": True,
        "old_and_new_training_sets_differ": True,
        "same_test_cases_fold_ids_labels": True,
        "external_validation": False,
        "release_pass": False,
    }
