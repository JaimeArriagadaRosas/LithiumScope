import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone

from lithiumscope.model_2.training.preprocessing.spectral_treatment import spectral_pipeline


def test_spectral_126_keeps_full_inference_contract_and_train_only_fit(tmp_path):
    import joblib
    x=pd.DataFrame(np.random.default_rng(42).normal(size=(40,170)),columns=[f"f{i}" for i in range(170)])
    y=np.tile([0,1],20)
    estimator=clone(spectral_pipeline(LogisticRegression(),"extra_trees","v3_126",42)).fit(x,y)
    assert len(estimator.named_steps["spectral_columns"].get_feature_names_out())==126
    test=x.iloc[:3].copy()
    baseline=estimator.predict_proba(test)
    test.iloc[:,126:]+=999
    assert np.allclose(estimator.predict_proba(test),baseline)
    with pytest.raises(ValueError):
        estimator.predict_proba(test.iloc[:,:126])
    path=tmp_path/"model.joblib"
    joblib.dump(estimator,path)
    assert np.allclose(joblib.load(path).predict_proba(test),baseline)


def test_kbest_selection_is_fitted_inside_pipeline():
    x=pd.DataFrame(np.random.default_rng(43).normal(size=(60,170)),columns=[f"f{i}" for i in range(170)])
    y=np.tile([0,1],30)
    estimator=spectral_pipeline(LogisticRegression(),"xgboost","kbest48",42).fit(x,y)
    assert estimator.named_steps["selection"].selector_.get_support().sum()==48
    assert estimator.predict_proba(x.iloc[:2]).shape==(2,2)
