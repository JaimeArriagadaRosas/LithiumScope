"""Deterministic M1 -> M2 decision analysis after independent inference and before rendering.

M1 and M2 inference remain independent. No true Li, model refitting, weights
or release labels influence the decision. True Li is optional retrospective QA.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TOP_K = 3
SCORE_TIE_ATOL = 1e-9


@dataclass(frozen=True)
class Decision:
    ranked: pd.DataFrame
    selected: pd.DataFrame
    winner_id: str | None
    status: str
    reason: str
    m1_rule: str
    m2_rule: str
    retrospective: dict


def _numeric(frame: pd.DataFrame, name: str) -> pd.Series:
    return pd.to_numeric(
        frame[name] if name in frame else pd.Series(np.nan, index=frame.index),
        errors="coerce",
    )


def calculate_decision(paired: pd.DataFrame) -> Decision:
    """Rank valid M1 estimates, prefer applicable rows and compare M2 scores.

    Domain warnings change the scope of the ranking, not numeric validity.
    Missing input, invalid predictions and missing scores remain excluded.
    """
    if paired is None or paired.empty:
        return Decision(
            pd.DataFrame(), pd.DataFrame(), None, "INSUFFICIENT",
            "No hay predicciones emparejadas.", "", "", {},
        )
    required = {
        "case_id", "Li_icpms_predicted", "prospectivity_score",
        "model_1_applicability_warning", "model_2_applicability_warning",
    }
    if not required <= set(paired):
        return Decision(
            paired.copy(), pd.DataFrame(), None, "INSUFFICIENT",
            "Faltan predicciones o diagnósticos de aplicabilidad.", "", "", {},
        )
    frame = paired.copy()
    if frame["case_id"].isna().any():
        raise ValueError("Report decision requires nonempty case IDs")
    frame["case_id"] = frame["case_id"].astype(str).str.strip()
    if frame.case_id.eq("").any() or frame.case_id.duplicated().any():
        raise ValueError("Report decision requires unique paired case_id")
    frame["_m1_ppm"] = _numeric(frame, "Li_icpms_predicted")
    frame["_m2_score"] = _numeric(frame, "prospectivity_score")
    m1_status = frame.model_1_applicability_warning.astype(str).str.strip().str.upper()
    allowed = {"OK", "OUT_OF_DOMAIN", "INSUFFICIENT_PROFILE"}
    frame["_m1_valid"] = (m1_status.isin(allowed) & np.isfinite(frame["_m1_ppm"])
                          & frame["_m1_ppm"].ge(0))
    frame["_m1_ok"] = frame["_m1_valid"] & m1_status.eq("OK")
    lower = _numeric(frame, "Li_icpms_interval_low_q90")
    # Apply the same ranking quantity to all usable M1 rows, including OOD.
    use_lower = (frame["_m1_valid"].any()
                 and np.isfinite(lower[frame["_m1_valid"]]).all())
    if use_lower:
        frame["_m1_rank_score"] = lower.clip(lower=0.0)
        m1_rule = "cota inferior q90 M1 (max(0, Li estimado - error absoluto q90))"
    else:
        frame["_m1_rank_score"] = frame["_m1_ppm"]
        m1_rule = "Li estimado M1 (intervalos q90 no disponibles para todos)"
    m2_status = frame.model_2_applicability_warning.astype(str).str.strip().str.upper()
    frame["_m2_valid"] = (m2_status.isin(allowed) & np.isfinite(frame["_m2_score"])
                          & frame["_m2_score"].between(0, 1))
    frame["_m2_ok"] = frame["_m2_valid"] & m2_status.eq("OK")
    frame["_selected"] = False
    frame["_winner"] = False
    frame = frame.sort_values(
        ["_m1_valid", "_m1_ok", "_m1_rank_score", "_m1_ppm", "case_id"],
        ascending=[False, False, False, False, True],
        kind="stable",
    ).reset_index(drop=True)
    frame["_m1_position"] = np.arange(1, len(frame) + 1)

    valid_m1 = frame[frame["_m1_valid"]]
    selected_ids = valid_m1.head(TOP_K).case_id.tolist()
    frame.loc[frame.case_id.isin(selected_ids), "_selected"] = True
    chosen = frame[frame["_selected"]].copy()
    applicable_m2 = chosen[chosen["_m2_ok"]]
    usable_m2 = chosen[chosen["_m2_valid"]]
    # Keep the original strict preference when at least two scores are
    # applicable. Otherwise a finite spectral ranking is exploratory only.
    pool = applicable_m2 if len(applicable_m2) >= 2 else usable_m2
    pool = pool.sort_values(["_m2_score", "case_id"], ascending=[False, True], kind="stable")
    winner_id: str | None = None
    status = "ABSTAIN"
    if len(chosen) == 0:
        status = "INSUFFICIENT"
        reason = "No hay predicciones M1 utilizables."
    elif len(pool) < 2:
        reason = "Menos de dos scores M2 válidos; comparación sin ganador."
    elif abs(float(pool["_m2_score"].iloc[0]) - float(pool["_m2_score"].iloc[1])) <= SCORE_TIE_ATOL:
        reason = "Empate espectral en primer lugar."
    else:
        winner_id = str(pool.case_id.iloc[0])
        frame.loc[frame.case_id == winner_id, "_winner"] = True
        if len(chosen) < TOP_K or not chosen["_m1_ok"].all() or len(applicable_m2) < 2:
            status = "EXPLORATORY"
            reason = "Prioridad exploratoria según los scores disponibles."
        elif len(applicable_m2) == TOP_K:
            status = "COMPARABLE"
            reason = "Tres evaluaciones M2 aplicables; preferencia relativa."
        else:
            status = "PROVISIONAL"
            reason = "Prioridad provisional entre scores M2 aplicables."

    # Ground truth is used AFTER the decision only for retrospective reporting.
    retrospective: dict = {}
    top = frame[frame["_selected"]]
    actual_top = _numeric(top, "Li_icpms")
    if len(top) == TOP_K and np.isfinite(actual_top).all():
        maximum = float(actual_top.max())
        true_best = top.loc[
            np.isclose(actual_top, maximum, rtol=0, atol=1e-9), "case_id"
        ].tolist()
        if len(true_best) == 1:
            retrospective = {
                "best_observed_in_top3": true_best[0],
                "m1_first_matches_observed": (
                    str(valid_m1.iloc[0].case_id) == true_best[0]
                ),
                "m2_winner_matches_observed": (
                    winner_id == true_best[0] if winner_id else None
                ),
                "not_used_for_selection": True,
            }
    chosen = frame[frame["_selected"]].copy()
    return Decision(
        frame, chosen, winner_id, status, reason, m1_rule,
        "Mayor score M2 aplicable si hay al menos dos; en otro caso, ranking exploratorio de scores válidos",
        retrospective,
    )
