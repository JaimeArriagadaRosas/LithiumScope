"""Spectral redundancy, scene/period and local-quality evidence, descriptive only."""
from __future__ import annotations

from hashlib import sha256
import numpy as np
import pandas as pd

from lithiumscope.core.scientific.eda_statistics import cohort_summary, feature_profiles, numeric_correlations


def redundancy_table(frame, features):
    buckets={}
    records=[]
    for name in features:
        values=pd.to_numeric(frame[name],errors="coerce").astype(float)
        finite=values[np.isfinite(values)]
        # Canonicalize signed zero and nonfinite values before hashing.
        clean=values.where(np.isfinite(values)).mask(values.eq(0),0.)
        digest=sha256(pd.util.hash_pandas_object(clean,index=False).to_numpy(dtype="<u8").tobytes()).hexdigest()
        candidates=buckets.setdefault(digest,[])
        for other,previous in candidates:
            if clean.equals(previous):
                records.append({"feature":name,"reference":other,"state":"identical_numeric_and_missingness",
                                "finite_rows":len(finite)})
                break
        else:
            candidates.append((name,clean))
            if finite.nunique()<=1:
                records.append({"feature":name,"reference":None,"state":"constant_or_all_missing",
                                "finite_rows":len(finite)})
    return pd.DataFrame(records,columns=["feature","reference","state","finite_rows"])


def period_summary(frame,target):
    if "sentinel_datetime" not in frame:
        return pd.DataFrame(columns=["period_utc","samples","target_median","mean_local_valid_fraction"])
    dates=pd.to_datetime(frame.sentinel_datetime,errors="coerce",utc=True,format="mixed")
    work=frame.assign(period_utc=dates.dt.strftime("%Y-%m").fillna("unknown"))
    records=[]
    for period,group in work.groupby("period_utc"):
        qa=pd.to_numeric(group.get("sixband_joint_valid_fraction",pd.Series(np.nan,index=group.index)),errors="coerce")
        records.append({"period_utc":period,"samples":len(group),
                        "target_median":pd.to_numeric(group[target],errors="coerce").median(),
                        "mean_local_valid_fraction":qa.mean()})
    return pd.DataFrame(records)


def descriptive_tables(frame,features,target):
    return {"feature_profiles":feature_profiles(frame,features,stage="prepared"),
            "cohort_summary":cohort_summary(frame,target),
            "spearman_correlations":numeric_correlations(frame,[*features,target]),
            "spectral_redundancy":redundancy_table(frame,features),
            "temporal_profiles":period_summary(frame,target)}
