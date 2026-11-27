"""Frozen OASIS-like cohort selection; no estimator fit or experiment execution."""
from __future__ import annotations
import pandas as pd
from lithiumscope.core.config import load_config
from lithiumscope.experiments.model_1.treatments.balancing import MIN_COHORT_ROWS
from lithiumscope.model_1.preparation.training_data import prepare_training_data
from lithiumscope.model_1.preparation.detection_limits import clean_detection_limits
from lithiumscope.datasets.validation.tabular.missing_values import normalize_missing_values
from lithiumscope.model_1.preparation.target_filtering import filter_target

def cohort(source):
    prepared=prepare_training_data(source)
    cfg=load_config("model_1")["data"]
    raw=normalize_missing_values(clean_detection_limits(prepared.raw_frame,
                                                       version=int(cfg["preparation_version"])))
    valid,target=filter_target(raw,candidates=list(cfg["target_candidates"]),
        lower_quantile=.025,upper_quantile=.975,zero_policy=cfg.get("target_zero_policy","retain"))
    lo,hi=pd.to_numeric(valid[target]).quantile([.025,.975])
    # Compute the target range before QC, as documented by OASIS. Apply to the
    # rows that pass our fixed QC; this defines a separate descriptive cohort.
    keep=prepared.y.between(lo,hi)
    if int(keep.sum())<MIN_COHORT_ROWS:
        raise ValueError("OASIS-like cohort below the 2043 real-row minimum")
    return prepared,keep,{"q025":float(lo),"q975":float(hi),"valid_target_before_qc":len(valid),
                          "rows_after_qc_and_trimming":int(keep.sum())}


