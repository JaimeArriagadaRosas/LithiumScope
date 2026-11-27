"""Prepare scientific report inputs; never fit, predict, or change source frames."""
from __future__ import annotations

from copy import deepcopy
import math

import numpy as np
import pandas as pd


from lithiumscope.core.scientific.report_values import operating_classification, physical_interval


def prepare_cases(model_1,model_2,paired,threshold):
    one,two,joined=(frame.copy() for frame in (model_1,model_2,paired))
    def column(frame,name):
        return frame.get(name,pd.Series(np.nan,index=frame.index))
    intervals=[physical_interval(lo,hi) for lo,hi in zip(
        column(one,"Li_icpms_interval_low_q90"),column(one,"Li_icpms_interval_high_q90"),strict=True)]
    one["report_interval_low"]=[p[0] for p in intervals]
    one["report_interval_high"]=[p[1] for p in intervals]
    two["report_operating_class"]=[operating_classification(s,t) for s,t in zip(
        column(two,"prospectivity_score"),column(two,"operating_threshold"),strict=True)]
    joined["report_absolute_error"]=(pd.to_numeric(column(joined,"Li_icpms"),errors="coerce")-
        pd.to_numeric(column(joined,"Li_icpms_predicted"),errors="coerce")).abs()
    joined["report_operating_class"]=[operating_classification(s,threshold) for s in column(joined,"prospectivity_score")]
    return one,two,joined


def prepare_selection_evidence(evidence):
    """Compute historical paired differences, with undefined zero-reference gains."""
    result=deepcopy(evidence)
    if not result or result.get("state")!="HISTORICAL_COMPARISON":
        return result
    def delta(a,b,relative=False):
        a,b=float(a),float(b)
        if not math.isfinite(a) or not math.isfinite(b) or (relative and a==0):
            return None
        return (a-b)/a if relative else b-a
    paired=result["model_1_paired"]
    shared=paired["common"]
    high=paired["high_lithium"]
    result["report_m1_gains"]={k:delta(shared["old"][k],shared["new"][k],True) for k in ("rmse","mae")}
    result["report_m1_gains"]["high_rmse"]=delta(high["old"]["rmse"],high["new"]["rmse"],True)
    compared={row["algorithm"]:row for row in result["model_2_same_706_comparison"]}
    old,new=compared["original_release_protocol_retrained"],compared["random_forest"]
    result["report_m2_deltas"]={key:delta(old[key],new[key]) for key in ("roc_auc","average_precision","brier")}
    return result


def _out_of_domain_count(frame: pd.DataFrame) -> int | str:
    if frame is None or frame.empty:
        return 0
    if "applicability_warning" in frame.columns:
        values = (
            frame["applicability_warning"]
            .astype(str)
            .str.upper()
        )
        return int((values == "OUT_OF_DOMAIN").sum())
    if "out_of_training_range_fraction" in frame.columns:
        values = pd.to_numeric(
            frame["out_of_training_range_fraction"],
            errors="coerce",
        )
        return int((values > 0.25).sum())
    return "N/D"



def report_statistics(model_1, model_2, concordance, threshold, diagnostics):
    m1 = (diagnostics or {}).get("out_of_domain_rows")
    if m1 is None:
        m1 = _out_of_domain_count(model_1)
    m2 = _out_of_domain_count(model_2)
    scores = pd.to_numeric(model_2.get("prospectivity_score", pd.Series(dtype=float)), errors="coerce")
    scores = scores[np.isfinite(scores)]
    counts = concordance["concordance"].value_counts().to_dict() if "concordance" in concordance else {}
    return {"m1_ood": m1, "m2_ood": m2, "score_count": len(scores),
            "positive_count": int((scores >= float(threshold)).sum()) if threshold is not None else None,
            "concordance_counts": counts}
