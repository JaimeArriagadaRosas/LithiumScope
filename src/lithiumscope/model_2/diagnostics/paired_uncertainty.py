"""Spatial-group paired uncertainty for experiments, never a release approval."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def _scores(labels, scores) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "brier": float(brier_score_loss(labels, scores)),
    }


def bootstrap_paired_groups(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    iterations: int = 500,
    seed: int = 42,
) -> dict:
    required = {"sample_index", "spatial_group", "y_true", "score"}
    if not (required <= set(reference) and required <= set(candidate)):
        raise ValueError("Missing spatial paired comparison columns")
    left = reference[list(required)].rename(columns={"score": "reference"})
    right = candidate[["sample_index", "y_true", "score"]].rename(
        columns={"y_true": "candidate_label", "score": "candidate"}
    )
    data = left.merge(right, on="sample_index", validate="one_to_one")
    if len(data) != len(reference) or len(data) != len(candidate):
        raise ValueError("Outer predictions do not cover the same test samples")
    if not (data.y_true == data.candidate_label).all():
        raise ValueError("Compared arms disagree on held-out labels")
    if data.sample_index.duplicated().any():
        raise ValueError("Repeated outer test sample")
    groups = data.spatial_group.astype(str).to_numpy()
    unique = np.unique(groups)
    if len(unique) < 5:
        raise ValueError("Insufficient independent geographic groups")
    blocks = [np.flatnonzero(groups == group) for group in unique]
    y = data.y_true.to_numpy(int)
    a = data.reference.to_numpy(float)
    b = data.candidate.to_numpy(float)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Nonfinite paired probabilities")
    old = _scores(y, a)
    new = _scores(y, b)
    dist = {key: [] for key in old}
    rng = np.random.default_rng(seed)
    for _ in range(iterations):
        sample = rng.integers(0, len(blocks), size=len(blocks))
        take = np.concatenate([blocks[int(i)] for i in sample])
        if len(np.unique(y[take])) != 2:
            continue
        scores_a = _scores(y[take], a[take])
        scores_b = _scores(y[take], b[take])
        for key in dist:
            dist[key].append(scores_b[key] - scores_a[key])
    if min(len(values) for values in dist.values()) < max(20, iterations // 2):
        raise ValueError("Too many single-class bootstrap draws")
    return {
        "clusters": len(unique),
        "paired_samples": len(data),
        "repetitions_requested": iterations,
        "successful_repetitions": len(dist["roc_auc"]),
        "metrics": {
            name: {
                "reference": old[name],
                "candidate": new[name],
                "candidate_minus_reference": new[name] - old[name],
                "ci95": np.quantile(values, [0.025, 0.975]).tolist(),
                "ci_includes_zero": bool(
                    min(np.quantile(values, [0.025, 0.975])) <= 0
                    <= max(np.quantile(values, [0.025, 0.975]))
                ),
            }
            for name, values in dist.items()
        },
        "scope": "exploratory_spatial_group_bootstrap_not_external_holdout",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--iterations", type=int, default=500)
    args = parser.parse_args()
    root = args.root
    regular = pd.read_csv(root / "paired_outer_predictions.csv")
    ablation = pd.read_csv(root / "data_ablation/paired_outer_predictions.csv")
    result = {}
    for calibrated in ("raw", "inner_sigmoid"):
        filtered = regular[regular.calibration == calibrated]
        result[f"regularized_vs_original_{calibrated}"] = bootstrap_paired_groups(
            filtered[filtered.arm == "frozen_rf_configuration"],
            filtered[filtered.arm == "regularized_depth8_leaf10"],
            iterations=args.iterations,
        )
    for arm in ("historical_only_qa95_train",
                "conflicting_sites_excluded_from_train"):
        result[f"{arm}_vs_all_qa95"] = bootstrap_paired_groups(
            ablation[ablation.arm == "all_qa95_train"],
            ablation[ablation.arm == arm],
            iterations=args.iterations,
        )
    (root / "paired_uncertainty.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for label, evidence in result.items():
        print(label, {
            key: (
                round(value["candidate_minus_reference"], 4),
                [round(v, 4) for v in value["ci95"]],
            )
            for key, value in evidence["metrics"].items()
        })


if __name__ == "__main__":
    main()
