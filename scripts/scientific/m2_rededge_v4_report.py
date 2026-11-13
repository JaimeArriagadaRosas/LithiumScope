"""Release-protocol and historical three-zone comparisons for completed v4 OOF.

All comparisons are development evidence. No external holdout or promotion.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

from lithiumscope.model_2.methodology.three_zone_ranking import ranking_evaluation


def _m(y, p):
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "ap": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
    }


def paired_group_ci(y, baseline, candidate, groups, *, seed=20261113, n=400):
    """Paired cluster-bootstrap exploratory intervals, resampling complete groups."""
    y = np.asarray(y, int)
    old = np.asarray(baseline, float)
    new = np.asarray(candidate, float)
    g = np.asarray(groups, str)
    unique = np.unique(g)
    locations = {key: np.flatnonzero(g == key) for key in unique}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        ix = np.concatenate([locations[key] for key in sampled])
        if len(np.unique(y[ix])) < 2:
            continue
        m0 = _m(y[ix], old[ix])
        m1 = _m(y[ix], new[ix])
        vals.append([m1[k] - m0[k] for k in ("roc_auc", "ap", "brier")])
    if len(vals) < n // 2:
        raise ValueError("Insufficient bootstrap replicates with both classes")
    arr = np.asarray(vals)
    return {
        k: np.quantile(arr[:, j], [0.025, 0.975]).round(8).tolist()
        for j, k in enumerate(("delta_auc_ci95", "delta_ap_ci95", "delta_brier_ci95"))
    }


def write_release_comparison(output: Path, prior: Path, algorithms: tuple[str, ...], manifest: pd.DataFrame) -> dict:
    old_cases = pd.read_csv(prior / "tables" / "old_release_protocol_paired_706.csv")
    reference = (
        old_cases.loc[old_cases.arm.eq("original_release_protocol_retrained")]
        .sort_values("test_index")
        .reset_index(drop=True)
    )
    if (
        len(reference) != 706
        or not np.array_equal(reference.test_index.to_numpy(int), np.arange(706))
        or not np.allclose(reference.li_ppm.to_numpy(float), manifest.Li_icpms.to_numpy(float))
    ):
        raise ValueError("Original CatBoost protocol comparison not aligned")
    labels = reference.y_true.to_numpy(int)
    expected = (manifest.Li_icpms.to_numpy(float) >= 21.6).astype(int)
    if not np.array_equal(labels, expected):
        raise ValueError("Original 21.6-ppm comparison labels drifted")
    old = reference.score.to_numpy(float)
    oldm = _m(labels, old)
    reports = []
    ranked = []
    for alg in algorithms:
        score = np.load(output / f"oof_{alg}.npy", allow_pickle=False)
        newm = _m(labels, score)
        uncertainty = paired_group_ci(
            labels,
            old,
            score,
            manifest.spatial_group.to_numpy(str),
        )
        reports.append(
            {
                "algorithm": alg,
                "baseline": "original_release_protocol_retrained_706",
                "release_auc": oldm["roc_auc"],
                "v4_auc": newm["roc_auc"],
                "delta_auc": newm["roc_auc"] - oldm["roc_auc"],
                "release_ap": oldm["ap"],
                "v4_ap": newm["ap"],
                "delta_ap": newm["ap"] - oldm["ap"],
                "release_brier": oldm["brier"],
                "v4_brier": newm["brier"],
                "delta_brier": newm["brier"] - oldm["brier"],
                **{key: json.dumps(val) for key, val in uncertainty.items()},
            }
        )
        view = reference[["test_index", "fold", "sample_id", "site_screening_id", "li_ppm", "y_true"]].copy()
        view["arm"] = alg
        view["score"] = score
        ranked.append(view)
    table = pd.DataFrame(reports)
    table.to_csv(output / "v4_vs_reconstructed_release_same_706.csv", index=False)
    summary, cases = ranking_evaluation(pd.concat(ranked, ignore_index=True))
    (output / "three_zone_v4_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    cases.to_csv(output / "three_zone_v4_cases.csv", index=False)
    return {"release_metric_reference": oldm, "three_zone": summary, "rows": len(table)}
