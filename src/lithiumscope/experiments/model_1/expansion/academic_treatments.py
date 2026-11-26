"""Bounded academic HGB treatment profiles. No target or feature fit on TEST."""
from __future__ import annotations

import numpy as np
import pandas as pd
from lithiumscope.experiments.model_1.expansion.academic_bridge import PROFILES
from lithiumscope.model_1.training.factory import build_pipeline
from lithiumscope.model_1.training.preprocessing.tabular import FeatureSchema
from lithiumscope.resources.manager import estimator_thread_scope

BASE="median_control"
# Native missing values, finer leaves, historical tail weights, and attenuated
# candidate transfer. Fixed before evaluating this campaign.
PROFILES_HGB={
    BASE: (False,10,1.,0.),
    "native_missing": (True,10,1.,0.),
    "native_fine": (True,5,1.,0.),
    "native_tail_2": (True,5,2.,0.),
    "native_tail_4": (True,5,4.,0.),
    "native_tail_2_transfer_010": (True,5,2.,.1),
    "native_tail_2_transfer_025": (True,5,2.,.25),
}


def training_weights(y, candidate, tail_multiplier, transfer_weight):
    if not y.index.equals(candidate.index) or not pd.api.types.is_bool_dtype(candidate) or candidate.isna().any():
        raise ValueError("Misaligned candidate mask")
    a=y.to_numpy(float)
    if not np.isfinite(a).all() or (a<0).any() or candidate.all():
        raise ValueError("Invalid historical TRAIN targets")
    if tail_multiplier not in (1.,2.,4.) or transfer_weight not in (0.,.1,.25):
        raise ValueError("Unregistered treatment weight")
    if candidate.any() and transfer_weight==0.:
        raise ValueError("Non-transfer arm must not contain candidates")
    q90=float(y.loc[~candidate].quantile(.9))
    weights=np.where(a>=q90,tail_multiplier,1.)*np.where(candidate,transfer_weight,1.)
    # Normalize historical weights only; do not undo transfer attenuation.
    weights=weights/weights[~candidate.to_numpy()].mean()
    return weights,q90


def fit_profile(x,y,candidate,schema,device,seed,profile):
    if not x.index.is_unique or not x.index.equals(y.index) or not x.index.equals(candidate.index):
        raise ValueError("TRAIN identity differs")
    native,leaf,tail,transfer=PROFILES_HGB[profile]
    params={k:v for k,v in PROFILES["hist_gradient_boosting"].items() if k!="treatment_policy"}
    params["min_samples_leaf"]=leaf
    model=build_pipeline("hist_gradient_boosting",device,seed,schema,params)
    if native:
        model.set_params(preprocess__numeric="passthrough")
    weights,q90=training_weights(y,candidate,tail,transfer)
    with estimator_thread_scope():
        model.fit(x,y,model__sample_weight=weights)
    def predict(frame):
        values=np.maximum(np.asarray(model.predict(frame),float),0.)
        if values.shape!=(len(frame),) or not np.isfinite(values).all():
            raise ValueError("Invalid HGB predictions")
        return values
    return predict,{"native_missing":native,"min_samples_leaf":leaf,"tail_multiplier":tail,
        "transfer_weight":transfer,"historical_q90":q90,"historical_rows":int((~candidate).sum()),
        "candidate_rows":int(candidate.sum()),"weight_min":float(weights.min()),"weight_max":float(weights.max()),
        "candidate_weight_sum":float(weights[candidate.to_numpy()].sum())}


def select_profile(scores):
    required={"profile","inner_fold","rmse","r2"}
    if not required.issubset(scores.columns) or set(scores.profile)!=set(PROFILES_HGB):
        raise ValueError("Incomplete inner scores")
    if scores.duplicated(["profile","inner_fold"]).any() or not np.isfinite(scores[["rmse","r2"]]).all().all():
        raise ValueError("Invalid inner scores")
    folds=set(scores.inner_fold)
    if len(folds)!=3 or any(set(g.inner_fold)!=folds for _,g in scores.groupby("profile")):
        raise ValueError("Incomplete inner folds")
    agg=scores.groupby("profile",sort=True)[["rmse","r2"]].mean().reset_index()
    agg["eligible"]=agg.r2.gt(.499)
    feasible=agg.loc[agg.eligible]
    if feasible.empty:
        return BASE,agg
    choice=feasible.sort_values(["rmse","r2","profile"],ascending=[True,False,True]).iloc[0]
    return str(choice.profile),agg
