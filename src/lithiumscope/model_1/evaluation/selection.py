"""Predeclared baseline benefit; relative ranking is not release approval."""
from __future__ import annotations

import numpy as np
import pandas as pd

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
    from lithiumscope.core.scientific.candidate_evidence import eligible_rows
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
