"""Versioned M1 treatment search. Only real TRAIN rows may be copied."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone


@dataclass(frozen=True)
class Treatment:
    resample: bool = False
    target: str = "identity"
    coordinates: bool = True
    robust: bool = False
    bounded: bool = False


POLICIES = {
    "nonnegative": Treatment(),
    "rare_group_sqrt": Treatment(True,"sqrt"),
    "bounded_log": Treatment(target="log1p",bounded=True),
    "bounded_log_no_coordinates": Treatment(target="log1p",coordinates=False,bounded=True),
    "rare_group_bounded_log": Treatment(True,"log1p",bounded=True),
    "rare_group_nonnegative": Treatment(True),
    "rare_group_no_coordinates": Treatment(True,coordinates=False),
    "robust_identity": Treatment(robust=True),
}
OPTIONS = {
    "random_forest": ("nonnegative","rare_group_sqrt"),
    "xgboost": ("nonnegative","rare_group_sqrt","bounded_log","bounded_log_no_coordinates"),
    "svm": ("nonnegative","bounded_log","bounded_log_no_coordinates"),
    "tabnet": ("nonnegative","robust_identity"),
    "hist_gradient_boosting": ("nonnegative","rare_group_sqrt","bounded_log","rare_group_bounded_log"),
    "catboost": ("nonnegative","rare_group_nonnegative","rare_group_no_coordinates"),
}
PROFILE = "refinement_v1"


def search_space(algorithm, base, profile):
    if profile=="legacy":
        return base
    if profile!=PROFILE or algorithm not in OPTIONS:
        raise ValueError("Unknown M1 treatment search profile")
    def space(trial):
        return {**base(trial),"treatment_policy":trial.suggest_categorical("treatment_policy",list(OPTIONS[algorithm]))}
    return space


def encode(values, target):
    y=np.asarray(values,float)
    if y.ndim!=1 or not len(y) or not np.isfinite(y).all() or (y<0).any():
        raise ValueError("Invalid M1 TRAIN target")
    return np.log1p(y) if target=="log1p" else np.sqrt(y) if target=="sqrt" else y


def decode(values, policy, train_max):
    z=np.asarray(values,float).reshape(-1)
    if not np.isfinite(z).all():
        raise ValueError("Nonfinite M1 latent prediction")
    negative=z<0
    limit=float(encode([train_max],policy.target)[0])
    upper=z>limit
    z=np.maximum(z,0.)
    if policy.bounded:
        z=np.minimum(z,limit)
    with np.errstate(over="raise",invalid="raise"):
        pred=np.expm1(z) if policy.target=="log1p" else z**2 if policy.target=="sqrt" else z
    if not np.isfinite(pred).all():
        raise ValueError("Nonfinite M1 decoded prediction")
    return pred,negative,upper


def rare_group_positions(y,groups,seed):
    """Same bounded rare-group bootstrap used in the validated experiments."""
    rare=np.flatnonzero(y.to_numpy(float)>=y.quantile(.90))
    pool=np.repeat(rare,4)
    weights=1/groups.iloc[pool].map(groups.iloc[rare].value_counts()).to_numpy(float)
    weights/=weights.sum()
    rng=np.random.default_rng(seed)
    return rng.choice(pool,size=min(len(pool),max(1,len(y)//4)),replace=False,p=weights)


class TreatedRegressor(RegressorMixin,BaseEstimator):
    requires_spatial_groups=True

    def __init__(self,regressor,policy_name,seed):
        self.regressor=regressor
        self.policy_name=policy_name
        self.seed=seed

    def fit(self,X,y,*,groups=None):
        policy=POLICIES[self.policy_name]
        if not isinstance(X,pd.DataFrame) or not X.index.is_unique:
            raise ValueError("Treatment requires unique indexed TRAIN rows")
        if groups is None or len(groups)!=len(X) or pd.isna(groups).any():
            raise ValueError("Treatment requires aligned spatial TRAIN groups")
        if isinstance(groups,pd.Series) and not groups.index.equals(X.index):
            raise ValueError("Misaligned TRAIN groups")
        if isinstance(y,pd.Series) and not y.index.equals(X.index):
            raise ValueError("Misaligned TRAIN targets")
        target=pd.Series(np.asarray(y,float),index=X.index)
        encode(target,policy.target)
        gg=pd.Series(np.asarray(groups),index=X.index)
        extra=rare_group_positions(target,gg,self.seed) if policy.resample else np.array([],dtype=int)
        positions=np.r_[np.arange(len(X)),extra]
        fit_x=X.iloc[positions].reset_index(drop=True)
        fit_y=pd.Series(encode(target.iloc[positions],policy.target),index=fit_x.index)
        fit_groups=gg.iloc[positions].reset_index(drop=True)
        self.train_max_=float(target.max())
        self.regressor_=clone(self.regressor)
        from lithiumscope.model_1.training.preprocessing.spatial_tabnet import fit_regression_estimator
        fit_regression_estimator(self.regressor_,fit_x,fit_y,fit_groups)
        self.treatment_evidence_={"policy":self.policy_name,"target_transform":policy.target,
            "coordinates":policy.coordinates,"robust":policy.robust,"bounded":policy.bounded,
            "real_train_rows":len(X),"fit_rows":len(fit_x),"train_target_max":self.train_max_,
            "copied_train_row_ids":[str(X.index[i]) for i in extra],
            "max_copies_per_parent":int(pd.Series(extra).value_counts().max()) if len(extra) else 0}
        return self

    @property
    def named_steps(self):
        return self.regressor_.named_steps

    @property
    def early_stopping_evidence_(self):
        return getattr(self.regressor_,"early_stopping_evidence_",None)

    def prediction_evidence(self,X):
        policy=POLICIES[self.policy_name]
        pred,negative,upper=decode(self.regressor_.predict(X),policy,self.train_max_)
        return pred,{"raw_negative_predictions":int(negative.sum()),
                     "above_train_support_predictions":int(upper.sum()),
                     "upper_bound_applied":int(upper.sum()) if policy.bounded else 0}

    def predict(self,X):
        return self.prediction_evidence(X)[0]


def treatment_pipeline(pipeline,schema,algorithm,policy_name,seed):
    if policy_name not in OPTIONS.get(algorithm,()):
        raise ValueError("Treatment is not allowed for this algorithm")
    policy=POLICIES[policy_name]
    from lithiumscope.model_1.training.preprocessing.tabular import FeatureSchema, build_preprocessor
    selected=FeatureSchema([c for c in schema.numeric if policy.coordinates or c not in {"Latitude","Longitude"}],
                           list(schema.categorical),schema.preparation_version,"stable_v1")
    pipeline.set_params(preprocess=build_preprocessor(selected,model_family=algorithm))
    if policy.robust:
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import RobustScaler
        pipeline.set_params(preprocess__numeric=Pipeline([
            ("imputer",SimpleImputer(strategy="median",keep_empty_features=True)),
            ("scale",RobustScaler())]))
    return TreatedRegressor(pipeline,policy_name,seed)
