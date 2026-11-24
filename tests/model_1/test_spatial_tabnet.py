from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from lithiumscope.model_1.training.algorithms.tabnet import TabNetRegressorAdapter
from lithiumscope.model_1.training.factory import build_pipeline
from lithiumscope.model_1.training.preprocessing.spatial_tabnet import (
    early_stopping_partition, fit_regression_estimator, early_stopping_evidence,
)
from lithiumscope.model_1.training.preprocessing.tabular import FeatureSchema
from lithiumscope.resources.detection.device import DeviceInfo


DEVICE=DeviceInfo("cpu","CPU",1,False,False,False)


def dataset():
    groups=pd.Series(np.repeat([f"g{i}" for i in range(8)],5),index=np.arange(100,140))
    x=pd.DataFrame({"x":np.arange(40,dtype=float),"rock":groups},index=groups.index)
    y=pd.Series(np.arange(1,41,dtype=float),index=groups.index)
    return x,y,groups


class FakeTabNet:
    def __init__(self, records):
        self.records=records
        self.best_epoch=2

    def fit(self,x,y,**kw):
        self.records.append((x.copy(),y.copy(),kw))
        self.mean=float(np.mean(y))

    def predict(self,x):
        return np.full((len(x),1),self.mean)


@pytest.mark.parametrize("transform",["identity","log1p"])
def test_tabnet_preprocess_excludes_spatial_validation_then_refits_all(monkeypatch,transform):
    x,y,g=dataset()
    records=[]
    monkeypatch.setattr(TabNetRegressorAdapter,"_new_model",lambda self:FakeTabNet(records))
    pipeline=build_pipeline("tabnet",DEVICE,42,FeatureSchema(["x"],["rock"],2),
                            {"max_epochs":8,"patience":2},target_transform=transform)
    estimator=fit_regression_estimator(clone(pipeline),x,y,g)
    evidence=early_stopping_evidence(estimator)
    tr,va=np.array(evidence["train_positions"]),np.array(evidence["validation_positions"])
    assert set(g.iloc[tr]).isdisjoint(g.iloc[va])
    assert sorted(np.r_[tr,va])==list(range(40))
    assert len(records)==2
    train,targets,kwargs=records[0]
    valid,valid_targets=kwargs["eval_set"][0]
    # Categories unique to spatial validation are unknown to early-stop encoder.
    assert train.shape[1]==1+g.iloc[tr].nunique()
    assert np.all(valid[:,1:]==0)
    assert train.shape[0]==len(tr) and valid.shape[0]==len(va)
    expected=np.log1p(y) if transform=="log1p" else y
    np.testing.assert_allclose(targets.ravel(),expected.iloc[tr],rtol=1e-6)
    np.testing.assert_allclose(valid_targets.ravel(),expected.iloc[va],rtol=1e-6)
    final,final_y,kw=records[1]
    assert final.shape==(40,9) and len(final_y)==40
    assert kw["max_epochs"]==3 and kw["patience"]==0 and "eval_set" not in kw
    assert kw["drop_last"] is False
    assert estimator.predict(x).shape==(40,)
    assert evidence["refit_n"]==40 and evidence["best_epoch"]==3


def test_tabnet_requires_aligned_nonempty_groups():
    x,y,g=dataset()
    for bad in (None,g.reset_index(drop=True),pd.Series("g",index=x.index),
                g.mask(g==g.iloc[0])):
        with pytest.raises(ValueError):
            early_stopping_partition(x,y,bad,seed=42,validation_fraction=.15)


def test_split_reproducible_and_not_selected_by_lithium():
    x,y,g=dataset()
    a=early_stopping_partition(x,y,g,seed=42,validation_fraction=.15)
    b=early_stopping_partition(x,y[::-1],g,seed=42,validation_fraction=.15)
    np.testing.assert_array_equal(a[0],b[0])
    np.testing.assert_array_equal(a[1],b[1])


def test_batches_never_drop_rows_or_leave_singleton():
    for n in [2,3,16,256,257,513,3420]:
        kw=TabNetRegressorAdapter._batch_policy(n)
        assert 2<=kw["batch_size"]<=n and n%kw["batch_size"]!=1
        assert not kw["drop_last"]


def test_optuna_passes_only_inner_train_groups(monkeypatch):
    from lithiumscope.model_1.training.optimization.optimizer import optimize_with_optuna
    from sklearn.model_selection import GroupKFold
    x,y,g=dataset()
    splits=list(GroupKFold(2).split(x,y,g))
    records=[]
    monkeypatch.setattr(TabNetRegressorAdapter,"_new_model",lambda self:FakeTabNet(records))
    calls=[]
    from lithiumscope.model_1.training.preprocessing import spatial_tabnet
    original=spatial_tabnet.fit_regression_estimator
    def captured(est,xx,yy,gg=None):
        calls.append((xx.index.tolist(),gg.index.tolist(),set(gg)))
        return original(est,xx,yy,gg)
    monkeypatch.setattr(spatial_tabnet,"fit_regression_estimator",captured)
    def factory(params):
        return build_pipeline("tabnet",DEVICE,42,FeatureSchema(["x"],["rock"],2),
                              {"max_epochs":8,"patience":2})
    optimize_with_optuna(factory,lambda trial:{},x,y,splits,1,groups=g,pruning_enabled=False)
    assert len(calls)==2 and len(records)==4
    for (indices,group_indices,groups),(tr,va) in zip(calls,splits,strict=True):
        assert indices==group_indices==x.iloc[tr].index.tolist()
        assert groups.isdisjoint(set(g.iloc[va]))
