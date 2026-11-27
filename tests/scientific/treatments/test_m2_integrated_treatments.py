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


def test_calibrated_svm_preserves_fold_local_spectral_treatment():
    from types import SimpleNamespace
    from sklearn.model_selection import GroupKFold
    from lithiumscope.model_2.training.factory import create_model
    from lithiumscope.model_2.training.preprocessing.spectral_treatment import fitted_spectral_evidence
    x=pd.DataFrame(np.random.default_rng(123).normal(size=(60,170)),columns=[f"f{i}" for i in range(170)])
    y=np.tile([0,1],30)
    groups=np.repeat(np.arange(6),10)
    folds=list(GroupKFold(3).split(x,y,groups))
    estimator=create_model("svm_rbf",SimpleNamespace(accelerator="cpu"),42,
        {"spectral_treatment":"v3_126","C":.5,"class_weight":None},
        calibrated=True,calibration_cv=folds)
    estimator.fit(x,y)
    assert estimator.predict_proba(x.iloc[:4]).shape==(4,2)
    assert fitted_spectral_evidence(estimator,"v3_126")["fitted_estimators"][0]["selected_count"]==126
