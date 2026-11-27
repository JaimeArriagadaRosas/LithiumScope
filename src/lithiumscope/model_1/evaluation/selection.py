"""Predeclared baseline benefit; relative ranking is not release approval."""
from __future__ import annotations

import numpy as np
import pandas as pd
from lithiumscope.core.scientific.candidate_evidence import eligible_rows

POLICY="mean_outer_rmse_strictly_better_than_both_train_references_v1"


def baseline_eligibility(ranking, *, preflight_ok, minimum_gain=0.0):
    if not np.isfinite(minimum_gain) or minimum_gain<0:
        raise ValueError("Invalid minimum baseline gain")
    result=ranking.copy()
    baseline=result.loc[result.is_baseline.eq(True)].set_index("algorithm")  # noqa: E712
    if not baseline.index.is_unique or set(baseline.index)!={"baseline_mean","baseline_median"}:
        raise ValueError("Both TRAIN-only statistical baselines are mandatory")
    references=baseline["rmse_mean"].astype(float)
    if not np.isfinite(references).all() or (references<0).any():
        raise ValueError("Undefined baseline RMSE")
    best=float(references.min())
    result["baseline_policy"]=POLICY
    result["best_baseline_rmse_mean"]=best
    result["rmse_gain_over_best_baseline"]=best-pd.to_numeric(result.rmse_mean,errors="coerce")
    result["baseline_gate_pass"]=False
    result["scientific_candidate_eligible"]=False
    result["release_eligible"]=False
    result["scientific_candidate_reason"]="reference_or_failed"
    valid=eligible_rows(result)&~result.is_baseline.eq(True)  # noqa: E712
    for idx in result.index[valid]:
        gain=result.at[idx,"rmse_gain_over_best_baseline"]
        passes=bool(np.isfinite(gain) and gain>minimum_gain)
        result.at[idx,"baseline_gate_pass"]=passes
        result.at[idx,"scientific_candidate_eligible"]=bool(passes and preflight_ok)
        result.at[idx,"scientific_candidate_reason"]=(
            "development_candidate_only" if passes and preflight_ok else
            "preflight_not_accepted" if not preflight_ok else "no_predeclared_baseline_benefit")
    return result


def ranking_row(result) -> dict:
    folds = result.fold_table
    return {
        "algorithm": (
            result.variant_id
            or result.algorithm
        ),
        "base_algorithm": result.algorithm,
        "target_transform": result.target_transform,
        "final_treatment_policy": result.final_params.get("treatment_policy","legacy"),
        "label": result.label,
        "rmse_mean": float(folds["rmse"].mean(skipna=False)),
        **{f"oof_{name}": value for name,value in result.overall_metrics.items()},
        "rmse_std": float(folds["rmse"].std(ddof=0)),
        "mae_mean": float(folds["mae"].mean()),
        "mae_std": float(folds["mae"].std(ddof=0)),
        "r2_mean": float(folds["r2"].mean()),
        "r2_std": float(folds["r2"].std(ddof=0)),
        "median_ae_mean": float(folds["median_ae"].mean()),
        "explained_variance_mean": float(folds["explained_variance"].mean()),
        "elapsed_seconds": result.elapsed_seconds,
        "status": "ok",
        "is_baseline": False,
    }


def rank_candidates(rows: list[dict]) -> pd.DataFrame:
    ranking = pd.DataFrame(rows)
    if ranking.empty:
        return ranking

    baseline = ranking[ranking.get("is_baseline", False) == True].copy()  # noqa: E712
    candidates = ranking[
        eligible_rows(ranking)
        & (ranking.get("is_baseline", False) != True)  # noqa: E712
    ].copy()
    failed = ranking[(~eligible_rows(ranking)) & (ranking["is_baseline"] != True)].copy()

    if not candidates.empty:
        candidates = candidates.sort_values(
            ["rmse_mean", "mae_mean", "r2_mean", "algorithm"],
            ascending=[True, True, False, True],
        ).reset_index(drop=True)
        candidates.insert(0, "rank", range(1, len(candidates) + 1))

    if not baseline.empty:
        baseline.insert(0, "rank", pd.NA)

    return pd.concat([candidates, baseline, failed], ignore_index=True, sort=False)
