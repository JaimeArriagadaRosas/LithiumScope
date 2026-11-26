"""Research-only attenuation of high-domain samples, including all bootstrap copies."""
from __future__ import annotations

import numpy as np
import pandas as pd

from lithiumscope.model_1.training.factory import build_pipeline
from lithiumscope.model_1.training.preprocessing.tabular import FeatureSchema
from lithiumscope.model_1.training.preprocessing.treatment_policy import POLICIES, encode, decode, rare_group_positions
from lithiumscope.resources.manager import estimator_thread_scope


def weighted_positions(y, groups, high, policy, seed, weight):
    if not (y.index.equals(groups.index) and y.index.equals(high.index)):
        raise ValueError("TRAIN weights and row identities differ")
    if weight not in (.1,.25,.5,1.):
        raise ValueError("Unregistered transfer weight")
    if not pd.api.types.is_bool_dtype(high) or high.isna().any():
        raise ValueError("Invalid transfer mask")
    extra=rare_group_positions(y,groups,seed) if policy.resample else np.array([],dtype=int)
    positions=np.r_[np.arange(len(y)),extra]
    weights=np.where(high.to_numpy(bool)[positions],weight,1.)
    return positions,weights


def fit_weighted(device,seed,params,x,y,groups,numeric,categorical,high,weight):
    if not (x.index.is_unique and x.index.equals(y.index) and x.index.equals(groups.index)):
        raise ValueError("Invalid indexed TRAIN data")
    policy_name=params["treatment_policy"]
    if policy_name not in ("nonnegative","rare_group_sqrt"):
        raise ValueError("Unexpected archived RF treatment")
    policy=POLICIES[policy_name]
    positions,weights=weighted_positions(y,groups,high,policy,seed,weight)
    schema=FeatureSchema(list(numeric),list(categorical),2,"stable_v1")
    model=build_pipeline("random_forest",device,seed,schema,
                         {k:v for k,v in params.items() if k!="treatment_policy"})
    fit_x=x.iloc[positions].reset_index(drop=True)
    target=encode(y.iloc[positions],policy.target)
    with estimator_thread_scope():
        model.fit(fit_x,target,model__sample_weight=weights)
    def predict(frame):
        return decode(model.predict(frame),policy,float(y.max()))[0]
    return predict,{"fit_rows":len(positions),"real_train_rows":len(x),
                    "target_transform":policy.target,"policy":policy_name,
                    "transfer_weight":weight,"transfer_fit_rows":int(high.iloc[positions].sum()),
                    "effective_transfer_weight":float(weights[high.iloc[positions].to_numpy(bool)].sum()),
                    "trace":[{"row":str(x.index[i]),"action":"bootstrap_copy"} for i in positions[len(x):]]}
