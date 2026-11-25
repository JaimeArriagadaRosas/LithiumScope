"""Versioned development eligibility. Never infer release approval from OOF."""
from __future__ import annotations

import numpy as np
import pandas as pd

from lithiumscope.model_1.evaluation.oof_evidence import regression_oof_table


POLICY = "paired_baseline_stability_and_high_li_v1"


def _summary(table):
    y = table.Li_icpms_real.to_numpy(float)
    p = table.Li_icpms_predicted.to_numpy(float)
    if not np.isfinite(np.c_[y, p]).all() or (y < 0).any():
        raise ValueError("Invalid OOF evidence")
    residual = p-y
    high = table.observed_high.to_numpy(bool)
    fold_rmse = table.assign(squared=residual**2).groupby("outer_fold").squared.mean().pow(.5)
    source_bias = (table.assign(error=residual).groupby("source_dataset", dropna=False).error.mean()
                   if "source_dataset" in table else pd.Series(dtype=float))
    return {"fold_rmse": fold_rmse, "high_n": int(high.sum()),
            "high_mae": float(np.abs(residual[high]).mean()) if high.any() else np.nan,
            "high_bias": float(residual[high].mean()) if high.any() else np.nan,
            "worst_source_abs_bias": float(source_bias.abs().max()) if len(source_bias) else np.nan,
            "negative_predictions": int((p < 0).sum())}


def apply_development_gate(ranking, prepared, results, baseline_oof, policy):
    """Use paired outer predictions and fixed relative limits; no TEST-based tuning.

    Limits are research-development guardrails, not validated geological tolerances.
    Missing required evidence prevents a candidate from passing the new gate.
    """
    limits = {k: float(policy[k]) for k in (
        "max_worst_fold_rmse_ratio", "min_fold_win_fraction", "max_high_mae_ratio",
        "max_high_abs_bias_ratio", "max_source_abs_bias_ratio")}
    if (not np.isfinite(list(limits.values())).all()
            or not 0 < limits["min_fold_win_fraction"] <= 1
            or any(v <= 0 for k, v in limits.items() if k != "min_fold_win_fraction")):
        raise ValueError("Invalid predeclared development gate")
    reference = {}
    for name in ("baseline_mean", "baseline_median"):
        part = baseline_oof.loc[baseline_oof.algorithm.eq(name)].sort_values("row_position")
        if len(part) != len(prepared.y) or not part.row_position.is_unique:
            raise ValueError("Incomplete paired baseline")
        reference[name] = (part, _summary(part))
    best = min(reference, key=lambda n: reference[n][1]["fold_rmse"].mean())
    ref, stats = reference[best]
    best_source = min((v[1]["worst_source_abs_bias"] for v in reference.values()
                       if np.isfinite(v[1]["worst_source_abs_bias"])), default=np.nan)
    best_high_mae = min(v[1]["high_mae"] for v in reference.values())
    best_high_bias = min(abs(v[1]["high_bias"]) for v in reference.values())
    result = ranking.copy()
    result["development_policy"] = POLICY
    result["development_reference"] = best
    result["development_gate_pass"] = False
    result["development_gate_reason"] = "baseline_gate_not_passed"
    for idx in result.index[result.scientific_candidate_eligible]:
        name = result.at[idx, "algorithm"]
        if name not in results:
            result.at[idx, "scientific_candidate_eligible"] = False
            result.at[idx, "development_gate_reason"] = "candidate_evidence_missing"
            continue
        candidate, data = results[name]
        table = regression_oof_table(data, candidate.predictions, candidate.outer_splits).sort_values("row_position")
        cols = ["row_position", "outer_fold", "Li_icpms_real", "fold_train_q75_ppm", "observed_high"]
        if not np.array_equal(table[cols].to_numpy(), ref[cols].to_numpy()):
            raise ValueError("Candidate and reference partitions/targets differ")
        for col in ("row_index", "source_dataset", "spatial_group"):
            if col in ref and (col not in table or not table[col].reset_index(drop=True).equals(ref[col].reset_index(drop=True))):
                raise ValueError("Candidate and reference observation identity differ")
        s = _summary(table)
        folds = s["fold_rmse"]
        reference_folds = stats["fold_rmse"]
        if not folds.index.equals(reference_folds.index):
            raise ValueError("Fold metrics do not align")
        win_fraction = float((folds < reference_folds).mean())
        checks = {
            "fold_wins": win_fraction >= limits["min_fold_win_fraction"],
            "worst_fold": folds.max() <= limits["max_worst_fold_rmse_ratio"] * reference_folds.max(),
            "high_mae": s["high_n"] > 0 and np.isfinite(best_high_mae)
                        and s["high_mae"] <= limits["max_high_mae_ratio"] * best_high_mae,
            "high_bias": np.isfinite(best_high_bias)
                         and abs(s["high_bias"]) <= limits["max_high_abs_bias_ratio"] * best_high_bias,
            "source_bias": np.isfinite(best_source) and np.isfinite(s["worst_source_abs_bias"])
                           and s["worst_source_abs_bias"] <= limits["max_source_abs_bias_ratio"] * best_source,
            "physical": s["negative_predictions"] == 0,
        }
        passed = all(checks.values())
        for key, value in {"fold_win_fraction": win_fraction, "worst_fold_rmse": float(folds.max()),
                           "high_li_support": s["high_n"], "high_li_mae": s["high_mae"],
                           "high_li_bias": s["high_bias"], "worst_source_abs_bias": s["worst_source_abs_bias"],
                           "negative_predictions": s["negative_predictions"]}.items():
            result.at[idx, key] = value
        result.at[idx, "development_gate_pass"] = passed
        result.at[idx, "scientific_candidate_eligible"] = passed
        reason = "development_candidate_only" if passed else ";".join(k for k, ok in checks.items() if not ok)
        result.at[idx, "development_gate_reason"] = reason
        result.at[idx, "scientific_candidate_reason"] = reason
    return result
