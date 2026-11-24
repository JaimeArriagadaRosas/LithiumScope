"""One-to-one M1 observation identity and fold-local reference labels."""
from __future__ import annotations

import numpy as np
import pandas as pd

from lithiumscope.core.scientific_checks import assert_oof_complete, assert_partition_coverage


def regression_oof_table(prepared, predictions, splits):
    n=len(prepared.y)
    assert_oof_complete(predictions,n)
    assert_partition_coverage(splits,n)
    if not prepared.y.index.is_unique or not prepared.x.index.equals(prepared.y.index):
        raise ValueError("M1 source row identity is invalid")
    fold_ids=np.zeros(n,dtype=int)
    thresholds=np.full(n,np.nan)
    truth=np.asarray(prepared.y,dtype=float)
    scores=np.asarray(predictions,dtype=float)
    if not np.isfinite(truth).all():
        raise ValueError("M1 truth is nonfinite")
    for i,(train,test) in enumerate(splits,1):
        fold_ids[test]=i
        thresholds[test]=np.quantile(truth[train],.75)
    frame=pd.DataFrame({"row_index":prepared.y.index,"row_position":np.arange(n),
                        "outer_fold":fold_ids,"Li_icpms_real":truth,
                        "Li_icpms_predicted":scores,"residual":truth-scores,
                        "fold_train_q75_ppm":thresholds,"observed_high":(truth>=thresholds).astype(int),
                        "predicted_high":(scores>=thresholds).astype(int)})
    source=getattr(prepared,"frame",None)
    if source is not None:
        if not source.index.equals(prepared.y.index):
            raise ValueError("M1 source metadata is not aligned")
        for name in ("source_dataset","source_sample","Sample","source_unique_id","Latitude","Longitude"):
            if name in source:
                frame[name]=source[name].to_numpy()
    groups=getattr(prepared,"groups",None)
    if groups is not None:
        if not groups.index.equals(prepared.y.index):
            raise ValueError("M1 spatial groups are not aligned")
        frame["spatial_group"]=groups.to_numpy()
    return frame
