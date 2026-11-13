"""Six-algorithm 706/783-row competition: fit-only QA regression guards.

Synthetic tests do not launch the costly complete Optuna competition.
"""
from __future__ import annotations

from pathlib import Path
import hashlib

import numpy as np
import pandas as pd
import pytest

from lithiumscope.model_2.training.quality import training_quality_mask
from lithiumscope.model_2.training import cv_runner
from lithiumscope.model_2.training.scientific_validation import (
    make_scientific_folds, scientific_labels
)


def test_quality_mask_keeps_expanded_cases_and_verifies_original_bytes(tmp_path):
    paths=[]
    hashcodes=[]
    for i in range(8):
        p=tmp_path/f"s{i}.npy"
        np.save(p,np.full((6,32,32),float(i)),allow_pickle=False)
        paths.append(str(p))
        hashcodes.append(hashlib.sha256(p.read_bytes()).hexdigest())
    manifest=tmp_path/"manifest.csv"
    pd.DataFrame({"sample_id":[f"s{i}" for i in range(8)],
        "image_path":paths,"Li_icpms":[10,12,40,30,25,20,13,23],
        "spatial_group":[f"g{i}" for i in range(8)]}).to_csv(manifest,index=False)
    inv=tmp_path/"qa.csv"
    pd.DataFrame({"manifest_index":list(range(8)),
        "sample_id":[f"s{i}" for i in range(8)],
        "image_sha256":hashcodes,
        "sixband_joint_valid_fraction":[1,.5,1,1,1,.91,.95,1]}).to_csv(inv,index=False)
    mask,summary=training_quality_mask(manifest,inventory_path=inv,
                                        minimum_joint_valid=.95)
    assert summary["manifest_rows"]==8
    assert summary["training_eligible"]==6
    assert summary["training_excluded"]==2
    assert mask.tolist()==[True,False,True,True,True,False,True,True]
    before=paths[0]
    arr=np.load(before)
    arr[0]=100
    np.save(before,arr,allow_pickle=False)
    with pytest.raises(ValueError,match="changed"):
        training_quality_mask(manifest,inventory_path=inv)


def test_six_algorithm_cv_training_quality_does_not_filter_outer_test(monkeypatch):
    n=60
    frame=pd.DataFrame({"red_mean":np.linspace(100,900,n),
                        "nir_mean":np.linspace(900,100,n)})
    li=np.asarray([5+15*(i%4) for i in range(n)],dtype=float)
    groups=pd.Series([f"site_{i//2}" for i in range(n)])
    folds=make_scientific_folds(
        pd.Series(li),groups,n_splits=5,random_state=42,quantile=.75)
    labels=pd.Series(scientific_labels(folds,n))
    quality=np.ones(n,dtype=bool)
    quality[[0,5,11,15,23,39]]=False
    observed=[]
    class StubEstimator:
        def predict_proba(self,x):
            score=.2+.0006*x["red_mean"].to_numpy(dtype=float)
            return np.column_stack((1-score,score))
    def fake_fit(algorithm,device,seed,params,x_train,y_train,
                 groups_train,inner_folds,**kw):
        assert len(x_train)==len(y_train)==len(groups_train)
        assert len(set(np.asarray(y_train)))==2
        observed.append((len(x_train),len(groups_train),algorithm))
        return StubEstimator()
    monkeypatch.setattr(cv_runner,"_fit_estimator",fake_fit)
    result=cv_runner.run_classification_cv(
        frame,labels,"catboost",None,42,5,groups,
        optimization={"enabled":False},inner_folds=3,
        scientific_folds=folds,lithium_values=li,
        quality_training_mask=quality)
    assert len(result.probabilities)==n
    assert np.isfinite(result.probabilities).all()
    assert len(observed)==5
    assert len(result.fold_table)==5
    assert result.fold_table.n_test.sum()==n
    for idx,fold in enumerate(folds):
        expected=int(quality[fold.train_idx].sum())
        assert observed[idx][0]==expected
        assert result.fold_table.n_train.iloc[idx]==expected
    assert any(
        index in np.concatenate([f.test_idx for f in folds])
        for index in np.flatnonzero(~quality)
    )
    # All held-out predictions, even six invalid TRAIN-quality source images.
    assert set(np.concatenate([f.test_idx for f in folds]))==set(range(n))


def test_historical_706_control_is_explicit_and_reversible():
    from lithiumscope.core.config import load_config
    cfg=load_config("model_2")
    assert cfg["training"]["expansion"]["enabled"] is False
    assert cfg["training"]["frozen_expected_rows"] == 706
    assert cfg["quality_training"]["policy"] == "train_only_preserve_all_outer_test"
    assert cfg["quality_training"]["enabled"] is True
    assert cfg["quality_training"]["minimum_joint_valid_fraction"]==.95
    assert cfg["optimization"]["enabled"] is True
    algorithms=cfg["model"]["algorithms"]
    assert algorithms==[
        "random_forest","extra_trees","hist_gradient_boosting",
        "xgboost","catboost","svm_rbf"
    ]



def test_optuna_receives_only_quality_train_rows_with_aligned_lithium(monkeypatch):
    # Simulate full nested optimizer wiring without running any expensive trials.
    n=60
    data=pd.DataFrame({"red_mean":np.arange(n,dtype=float)+100.,
                       "nir_mean":np.linspace(300,900,n)})
    lithium=np.asarray([5+15*(i%4) for i in range(n)],dtype=float)
    groups=pd.Series([f"g{i//2}" for i in range(n)])
    folds=make_scientific_folds(
        pd.Series(lithium),groups,n_splits=5,random_state=42,quantile=.75)
    labels=pd.Series(scientific_labels(folds,n))
    keep=np.ones(n,dtype=bool)
    keep[[2,5,9,13,44,59]]=False
    observations=[]
    def optimizer(factory,space,x,y,splits,*,lithium_values,**kwargs):
        assert len(x)==len(y)==len(lithium_values)
        assert len(splits)==3
        assert kwargs["n_trials"]==2
        # Every excluded row has a unique spectral red_mean; none should appear.
        assert not any(v in set(x.red_mean.to_numpy(float))
                       for v in data.red_mean.to_numpy(float)[~keep])
        observations.append(len(x))
        return {}
    class Predictor:
        def predict_proba(self,x):
            p=.2+.0004*x.red_mean.to_numpy(float)
            return np.column_stack([1-p,p])
    monkeypatch.setattr(cv_runner,"optimize_classifier_with_optuna",optimizer)
    monkeypatch.setattr(cv_runner,"_fit_estimator",lambda *a,**kw:Predictor())
    result=cv_runner.run_classification_cv(
        data,labels,"catboost",None,42,5,groups,
        optimization={"enabled":True,"n_trials":2},inner_folds=3,
        scientific_folds=folds,lithium_values=lithium,
        quality_training_mask=keep)
    assert len(observations)==6  # five outer folds and final full optimization
    assert observations[-1]==int(keep.sum())
    assert result.fold_table.n_test.sum()==n
