"""Descriptive source, target and predictor evidence for M1."""
from __future__ import annotations

import numpy as np
import pandas as pd

from lithiumscope.core.scientific.eda_statistics import (
    category_frequencies, cohort_summary, feature_profiles, numeric_correlations, source_series,
)


def target_tails(frame, target, *, stage):
    if target not in frame:
        return pd.DataFrame(columns=["stage","source","finite_nonnegative","negative","nonfinite_or_missing","q95","q99","maximum"])
    rows=[]
    for source,group in frame.groupby(source_series(frame),sort=True):
        y=pd.to_numeric(group[target],errors="coerce")
        finite=np.isfinite(y)
        valid=y[finite & y.ge(0)]
        rows.append({"stage":stage,"source":str(source),"finite_nonnegative":len(valid),
                     "negative":int((finite & y.lt(0)).sum()),"nonfinite_or_missing":int((~finite).sum()),
                     "q95":valid.quantile(.95),"q99":valid.quantile(.99),"maximum":valid.max()})
    return pd.DataFrame(rows)


def descriptive_tables(prepared):
    cols=[*prepared.schema.numeric,*prepared.schema.categorical]
    profiles=pd.concat([feature_profiles(prepared.raw_frame,cols,stage="raw"),
                        feature_profiles(prepared.frame,cols,stage="prepared")],ignore_index=True)
    categories=pd.concat([category_frequencies(prepared.raw_frame,prepared.schema.categorical).assign(stage="raw"),
                          category_frequencies(prepared.frame,prepared.schema.categorical).assign(stage="prepared")],
                         ignore_index=True)
    tails=pd.concat([target_tails(prepared.raw_frame,prepared.target,stage="raw"),
                     target_tails(prepared.frame,prepared.target,stage="prepared")],ignore_index=True)
    return {"feature_profiles":profiles,"category_frequencies":categories,
            "cohort_summary":cohort_summary(prepared.frame,prepared.target),
            "target_tails":tails,"spearman_correlations":numeric_correlations(
                prepared.frame,[*prepared.schema.numeric,prepared.target])}
