"""Deterministic M1 -> M2 decision analysis ONLY at report-render time.

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
    """Rank top 3 conservative M1 estimates, then valid M2 spectral scores.

    Fail closed when fewer than three M1 candidates are in domain; do not
    silently substitute model predictions with observed Li ground truth.
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
    frame["_m1_ok"] = (
        frame.model_1_applicability_warning.astype(str).str.upper().eq("OK")
        & np.isfinite(frame["_m1_ppm"])
        & frame["_m1_ppm"].ge(0)
    )
    lower = _numeric(frame, "Li_icpms_interval_low_q90")
    # Use a consistent rule across all in-domain candidate rows: if q90 is
    # unavailable for any, use predicted ppm for everyone.
    use_lower = (
        frame["_m1_ok"].any()
        and np.isfinite(lower[frame["_m1_ok"]]).all()
    )
    if use_lower:
        frame["_m1_rank_score"] = lower.clip(lower=0.0)
        m1_rule = "cota inferior q90 M1 (max(0, Li estimado - error absoluto q90))"
    else:
        frame["_m1_rank_score"] = frame["_m1_ppm"]
        m1_rule = "Li estimado M1 (intervalos q90 no disponibles para todos)"
    frame["_m2_ok"] = (
        frame.model_2_applicability_warning.astype(str).str.upper().eq("OK")
        & np.isfinite(frame["_m2_score"])
        & frame["_m2_score"].between(0, 1)
    )
    frame["_selected"] = False
    frame["_winner"] = False
    frame = frame.sort_values(
        ["_m1_ok", "_m1_rank_score", "_m1_ppm", "case_id"],
        ascending=[False, False, False, True],
        kind="stable",
    ).reset_index(drop=True)
    frame["_m1_position"] = np.arange(1, len(frame) + 1)

    valid_m1 = frame[frame["_m1_ok"]]
    if len(valid_m1) < TOP_K:
        return Decision(
            frame, pd.DataFrame(), None, "INSUFFICIENT",
            f"Solo {len(valid_m1)} candidatos M1 aplicables; se necesitan {TOP_K}.",
            m1_rule, "Mayor score espectral M2 dentro de los tres M1.", {},
        )
    selected_ids = valid_m1.head(TOP_K).case_id.tolist()
    frame.loc[frame.case_id.isin(selected_ids), "_selected"] = True
    chosen = frame[frame["_selected"]].copy()
    eligible_m2 = chosen[chosen["_m2_ok"]].sort_values(
        ["_m2_score", "case_id"], ascending=[False, True], kind="stable",
    )
    winner_id: str | None = None
    status = "ABSTAIN"
    if len(eligible_m2) < 2:
        reason = (
            "Menos de dos de las tres imágenes tienen score M2 "
            "aplicable; no se declara ganador."
        )
    elif (
        len(eligible_m2) > 1
        and abs(float(eligible_m2["_m2_score"].iloc[0])
                - float(eligible_m2["_m2_score"].iloc[1])) <= SCORE_TIE_ATOL
    ):
        reason = "Empate espectral en primer lugar; no se declara ganador."
    else:
        winner_id = str(eligible_m2.case_id.iloc[0])
        frame.loc[frame.case_id == winner_id, "_winner"] = True
        status = "COMPARABLE" if len(eligible_m2) == TOP_K else "PROVISIONAL"
        reason = (
            "Tres evaluaciones M2 dentro de dominio; preferencia relativa."
            if status == "COMPARABLE"
            else "Prioridad provisional: al menos un candidato M2 fuera de dominio."
        )

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
        "max score espectral M2; excluir valores no finitos y fuera de dominio",
        retrospective,
    )
