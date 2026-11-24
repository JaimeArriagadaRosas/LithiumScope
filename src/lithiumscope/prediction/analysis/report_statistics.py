"""Scientific summary values prepared before report rendering."""
from __future__ import annotations
import pandas as pd
import numpy as np

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
