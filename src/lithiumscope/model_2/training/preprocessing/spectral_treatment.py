"""Fold-local spectral ablation, preserving the verified full input contract."""
from __future__ import annotations

from functools import partial
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, VarianceThreshold, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROFILE="spectral_refinement_v1"
OPTIONS={
    "random_forest":("v4","v3_126"),
    "extra_trees":("v4","v3_126"),
    "hist_gradient_boosting":("v4",),
    "xgboost":("v4","kbest48"),
    "catboost":("v4","kbest48"),
    "svm_rbf":("v4","v3_126"),
}


class SpectralColumns(TransformerMixin,BaseEstimator):
    def __init__(self,representation="v4"):
        self.representation=representation

    def fit(self,X,y=None):
        if self.representation not in {"v4","v3_126","kbest48"}:
            raise ValueError("Unknown spectral treatment")
        if not hasattr(X,"columns") or X.shape[1]!=170 or not X.columns.is_unique:
            raise ValueError("Spectral treatment requires the verified ordered V4 matrix")
        self.feature_names_in_=np.asarray(X.columns,dtype=object)
        self.n_features_in_=170
        return self

    def transform(self,X):
        if not hasattr(X,"columns") or list(X.columns)!=list(self.feature_names_in_):
            raise ValueError("Spectral inference schema differs from fitted V4 contract")
        result=X.iloc[:,:126] if self.representation=="v3_126" else X
        if not np.isfinite(result.to_numpy(float)).all():
            raise ValueError("Nonfinite verified spectral descriptors")
        return result

    def get_feature_names_out(self,input_features=None):
        return self.feature_names_in_[:126] if self.representation=="v3_126" else self.feature_names_in_


class AdaptiveKBest(TransformerMixin,BaseEstimator):
    def __init__(self,seed=42,k=48):
        self.seed=seed
        self.k=k

    def fit(self,X,y):
        if X.shape[1]<1 or len(X)<4:
            raise ValueError("Insufficient TRAIN data for spectral selection")
        self.selector_=SelectKBest(partial(mutual_info_classif,random_state=self.seed),
                                    k=min(self.k,X.shape[1],len(X)-1)).fit(X,y)
        return self

    def transform(self,X):
        return self.selector_.transform(X)

    def get_feature_names_out(self,input_features=None):
        return self.selector_.get_feature_names_out(input_features)


def spectral_pipeline(estimator,algorithm,representation,seed):
    if representation not in OPTIONS.get(algorithm,()):
        raise ValueError("Spectral treatment not allowed for this family")
    steps=[("spectral_columns",SpectralColumns(representation)),
           ("impute",SimpleImputer(strategy="median",keep_empty_features=True)),
           ("variance",VarianceThreshold()),("scale",StandardScaler())]
    if representation=="kbest48":
        steps.append(("selection",AdaptiveKBest(seed)))
    steps.append(("classifier",estimator))
    return Pipeline(steps)


def refinement_search_space(algorithm,base):
    if algorithm not in OPTIONS:
        raise ValueError("Unknown spectral refinement family")

    def space(trial):
        # Every hyperparameter returned here is a real Optuna suggestion;
        # the persisted best_params fully reconstructs the chosen estimator.
        if algorithm=="hist_gradient_boosting":
            params={"learning_rate":trial.suggest_float("learning_rate",.01,.15,log=True),
                    "max_iter":trial.suggest_int("max_iter",150,450),
                    "max_leaf_nodes":trial.suggest_int("max_leaf_nodes",8,24),
                    "min_samples_leaf":trial.suggest_int("min_samples_leaf",15,40),
                    "l2_regularization":trial.suggest_float("l2_regularization",.1,10.,log=True)}
        elif algorithm=="xgboost":
            params={"n_estimators":trial.suggest_int("n_estimators",200,650),
                    "learning_rate":trial.suggest_float("learning_rate",.01,.15,log=True),
                    "max_depth":trial.suggest_int("max_depth",2,5),
                    "min_child_weight":trial.suggest_int("min_child_weight",3,12),
                    "reg_lambda":trial.suggest_float("reg_lambda",1.,20.,log=True),
                    "subsample":trial.suggest_float("subsample",.7,1.),
                    "colsample_bytree":trial.suggest_float("colsample_bytree",.6,1.)}
        elif algorithm=="catboost":
            params={"iterations":trial.suggest_int("iterations",250,650),
                    "depth":trial.suggest_int("depth",3,6),
                    "learning_rate":trial.suggest_float("learning_rate",.01,.15,log=True),
                    "l2_leaf_reg":trial.suggest_float("l2_leaf_reg",1.,20.,log=True)}
        elif algorithm=="svm_rbf":
            params={"C":trial.suggest_float("C",.1,5.,log=True),
                    "gamma":trial.suggest_categorical("gamma",["scale",.0001,.001,.01,.05]),
                    "class_weight":trial.suggest_categorical("class_weight",[None,"balanced"])}
        else:
            params=base(trial)
        return {**params,"spectral_treatment":trial.suggest_categorical("spectral_treatment",list(OPTIONS[algorithm]))}
    return space
