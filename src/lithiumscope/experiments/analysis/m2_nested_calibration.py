"""Spatially nested threshold/calibration for the RF pilot's best predeclared arm.

Calibration and threshold learned on outer TRAIN only, using inner GroupKFold OOF.
No external ten-case labels enter the fitting, selection or optimization path.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, brier_score_loss
from lithiumscope.core.paths import PROJECT_ROOT
from lithiumscope.experiments.analysis.m2_generalization_pilot import (
    OUTPUT, get_train_indices, load_contract, metrics, model_params,
    nested_decision, sha256,
)


def run(root: Path = PROJECT_ROOT) -> Path:
    c = load_contract(root)
    output = root / OUTPUT
    evidence = output / "paired_fold_predictions.csv"
    if not evidence.is_file():
        raise FileNotFoundError("Run the fixed six-arm --pilot first")
    df = pd.read_csv(evidence)
    arm = df.loc[df.arm == "full126_regularized"].copy()
    if len(arm) != 783 or arm.sample_index.nunique() != 783:
        raise ValueError("Incomplete RF regularized outer spatial OOF")
    cols = list(c["bundle"]["features"])
    manifest = c["manifest"]
    features = c["features"]
    params = model_params(c["bundle"], regularized=True)
    output_rows, threshold_rows = [], []
    for i, fold in enumerate(c["folds"], 1):
        training_indices = get_train_indices(
            c, fold, buffer5=False, historical_only=False
        )
        test_indices = np.asarray(fold["test_indices"], dtype=int)
        samples = arm[arm.fold == i].set_index("sample_index")
        if set(samples.index) != set(test_indices.tolist()):
            raise ValueError("Frozen OOF fold no longer matches test identities")
        li_cutoff = float(fold["label_threshold_ppm"])
        train_y = (manifest.Li_icpms.iloc[training_indices].to_numpy(float)
                   >= li_cutoff).astype(int)
        x_train = features.iloc[training_indices][cols]
        groups = manifest.spatial_group.iloc[training_indices]
        if set(groups) & set(manifest.spatial_group.iloc[test_indices]):
            raise ValueError("Outer held-out groups leaked into calibration")
        cal, threshold, detail = nested_decision(
            x_train, train_y, groups, params
        )
        scores = samples.loc[test_indices, "score"].to_numpy(float)
        calibrated = cal.predict_proba(scores.reshape(-1, 1))[:, 1]
        true = samples.loc[test_indices, "y_true_fold"].to_numpy(int)
        for k, index in enumerate(test_indices):
            output_rows.append({
                "sample_index": int(index), "fold": i,
                "sample_id": str(manifest.sample_id.iloc[index]),
                "y_true": int(true[k]), "score_raw": float(scores[k]),
                "score_calibrated_nested": float(calibrated[k]),
                "train_only_frozen_threshold": float(threshold),
                "outer_test_used_for_calibration": False,
                "outer_test_used_for_threshold_selection": False,
            })
        threshold_rows.append({"fold": i, "outer_test_count": len(test_indices),
                               **detail})
        print("NESTED_FOLD",i,"threshold",round(threshold,5),
              "outer_n",len(test_indices),flush=True)
    data = pd.DataFrame(output_rows).sort_values("sample_index")
    if data.sample_index.tolist() != list(range(783)):
        raise ValueError("Outer predictions not in the frozen 783-row order")
    y = data.y_true.to_numpy(int)
    raw = data.score_raw.to_numpy(float)
    cal = data.score_calibrated_nested.to_numpy(float)
    frozen = data.train_only_frozen_threshold.to_numpy(float)
    summary = {
        "method": "outer_group_fold_inner_3fold_calibration_and_threshold",
        "outer_test_used_for_optimization": False,
        "ten_external_cases_used_for_optimization": False,
        "historical_folds_unchanged": True,
        "new_model_or_release_saved": False,
        "raw_metrics_0p5": metrics(y, raw),
        "calibrated_metrics_0p5": metrics(y, cal),
        "nested_frozen_balanced_accuracy": float(
            balanced_accuracy_score(y, cal >= frozen)
        ),
        "nested_calibrated_brier": float(brier_score_loss(y, cal)),
        "threshold_min": float(np.min(frozen)),
        "threshold_max": float(np.max(frozen)),
        "thresholds_are_fold_specific_not_global_release_threshold": True,
        "traceability_model_sha256": sha256(c["paths"]["model"]),
    }
    data.to_csv(output / "nested_full126_predictions.csv", index=False)
    pd.DataFrame(threshold_rows).to_csv(
        output / "nested_full126_train_thresholds.csv", index=False
    )
    (output / "nested_full126_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("NESTED_SUMMARY",summary,flush=True)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.parse_args()
    run()
