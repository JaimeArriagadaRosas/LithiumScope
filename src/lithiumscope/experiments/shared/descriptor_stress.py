"""Local frozen spectral descriptors: common cohort robustness, not external accuracy."""
from __future__ import annotations

import numpy as np
import pandas as pd

from lithiumscope.core.config import load_config
from lithiumscope.datasets.management.diagnostic_provisioning import prepare_diagnostic_source
from lithiumscope.model_2.preparation.features.v4_inputs import load_spectral_v4_training_frame
from lithiumscope.model_2.schema import out_of_range_fraction
from lithiumscope.results.atomic_exports import write_table


def load_common_matrix():
    cfg=load_config("model_2")
    source=prepare_diagnostic_source("model_2")
    frame=load_spectral_v4_training_frame(source,cfg)
    manifest=pd.read_csv(source)
    return frame,manifest.sample_id.astype(str)


def descriptor_stress(bundle,frame,ids,destination):
    expected=list(bundle["features"])
    if not set(expected)<=set(frame.columns):
        raise ValueError("Frozen descriptor matrix lacks expected features")
    x=frame[expected].astype(float)
    if not np.isfinite(x.to_numpy()).all():
        raise ValueError("Nonfinite frozen descriptors")
    base=bundle["estimator"].predict_proba(x)[:,1]
    rows=[]
    for seed in range(5):
        rng=np.random.default_rng(seed)
        for intensity in (0.,.01,.05,.10,.25):
            sample=x.copy()
            if intensity:
                sample=sample*(1+rng.normal(0,intensity,size=sample.shape))
            scores=bundle["estimator"].predict_proba(sample)[:,1]
            if not np.isfinite(scores).all() or ((scores<0)|(scores>1)).any():
                raise ValueError("Invalid spectral descriptor scores")
            for i,score in enumerate(scores):
                rows.append({"sample_id":ids.iloc[i],"seed":seed,"scenario":f"descriptor_noise_{intensity:g}",
                             "score":float(score),"absolute_score_shift":float(abs(score-base[i]))})
    output=destination/"descriptor_stress.csv"
    write_table(output,pd.DataFrame(rows))
    return output
