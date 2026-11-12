"""Phase C control tests. Synthetic, fast and no actual model fitting."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lithiumscope.model_2.methodology import ablations as a
from lithiumscope.model_2.methodology.contracts import sample_contract


@pytest.fixture
def c():
    data=pd.DataFrame({
        "sample_id":list("ABCDEFGH"),
        "image_path":[f"patch{i}.npy" for i in range(8)],
        "Li_icpms":[2.,3.,12.,14.,5.,16.,7.,15.],
        "latitude":[0.,.02,.1,.2,.3,.4,.005,.22],
        "longitude":[-70.]*8,
        "spatial_group":[f"g{i}" for i in range(8)],
        "sentinel_scene_id":[f"s{i}" for i in range(8)],
    })
    return sample_contract(data)


@pytest.fixture
def f():
    # Historical count is overridden to 6 in tests; new rows 6 and 7
    return {"train_indices":[1,3,5,6],"test_indices":[0,2,4,7],
            "label_threshold_ppm":np.quantile([3.,14.,16.],.75)}


def test_historical_test_frozen_and_train_only_threshold(monkeypatch,c,f):
    monkeypatch.setattr(a,"HISTORICAL_N",6)
    train,old,test,threshold=a.define_outer_fold(f,c)
    assert train.tolist()==[1,3,5,6]
    assert old.tolist()==[1,3,5]
    assert test.tolist()==[0,2,4]
    assert threshold==pytest.approx(np.quantile([3,14,16],.75))
    bad=dict(f);bad["label_threshold_ppm"]=99
    with pytest.raises(ValueError,match="threshold"):
        a.define_outer_fold(bad,c)


def test_added_training_samples_not_in_held_out(monkeypatch,c,f):
    monkeypatch.setattr(a,"HISTORICAL_N",6)
    threshold=float(f["label_threshold_ppm"])
    train=a.select_train_indices("expanded_all",f,c,threshold=threshold)
    baseline=a.select_train_indices("legacy_all",f,c,threshold=threshold)
    assert train.tolist()==[1,3,5,6]
    assert baseline.tolist()==[1,3,5]
    assert 7 not in train and 7 not in baseline


def test_buffer_excludes_near_train_without_touching_test(monkeypatch,c,f):
    monkeypatch.setattr(a,"HISTORICAL_N",6)
    source=list(f["test_indices"])
    tr=a.select_train_indices("expanded_buffer_5km",f,c,
                              threshold=f["label_threshold_ppm"])
    assert tr.tolist()==[3,5]
    assert f["test_indices"]==source
    assert 6 not in tr and 1 not in tr


def test_drop_mixed_sites_detects_only_current_training_rows(monkeypatch,c,f):
    monkeypatch.setattr(a,"HISTORICAL_N",6)
    c=c.copy()
    # New record G conflicts with B, both in current TRAIN; absent in
    # reference test. Their site is intentionally identical.
    c.loc[6,"latitude"]=c.loc[1,"latitude"]
    c.loc[6,"Li_icpms"]=30.
    c=sample_contract(c)
    tr=a.select_train_indices("expanded_drop_mixed_train_sites",f,c,
                              threshold=f["label_threshold_ppm"])
    assert tr.tolist()==[3,5]
    # Modifying *test Li* must not change the selected training indices.
    c.loc[0,"Li_icpms"]=999.
    c=sample_contract(c)
    again=a.select_train_indices("expanded_drop_mixed_train_sites",f,c,
                                 threshold=f["label_threshold_ppm"])
    assert tr.tolist()==again.tolist()


def test_site_weights_are_inverse_site_frequency_and_normalized(c):
    c=c.copy()
    c.loc[1,"latitude"]=c.loc[0,"latitude"]
    c=sample_contract(c)
    w=a.site_weights(c,np.array([0,1,2,3]))
    assert w[0]==pytest.approx(w[1])
    assert w[2]>w[0]
    assert float(w.mean())==pytest.approx(1.0)


def test_metrics_common_binary_reference():
    result=a.score_metrics([0,0,1,1],[.15,.35,.65,.85])
    assert result["n"]==4
    assert result["auc"]==1.0
    assert result["ap"]==1.0
    assert result["recall_at_threshold"]==1.0
    assert result["evaluated_threshold"]==0.5
    custom=a.score_metrics([0,1,1],[.1,.2,.3],.25)
    assert custom["recall_at_threshold"]==.5
    assert custom["evaluated_threshold"]==.25


def test_bootstrap_uses_common_held_out_rows_and_groups():
    rows=[]
    for arm,sc in [
        ("legacy_all",[.2,.8,.1,.9,.3,.7]),
        ("expanded_all",[.1,.9,.05,.95,.2,.8])]:
        for i,(y,score) in enumerate(zip([0,1,0,1,0,1],sc)):
            rows.append({"arm":arm,"test_index":i,"y_true":y,
                         "score":score,"spatial_group":f"g{i//2}"})
    result=a.bootstrap_group_delta(pd.DataFrame(rows),
        "legacy_all","expanded_all",repeats=20,seed=1)
    assert result["auc_bootstrap_valid"]>0
    assert "ap_delta_ci95_high" in result


def test_arms_predeclared_and_not_assigned_hyperopt():
    assert len(a.ARMS)==8
    assert "legacy_all" in a.ARMS
    assert "expanded_buffer_5km" in a.ARMS
    assert "expanded_inverse_site_weights" in a.ARMS
    assert "expanded_exclude_shared_scenes" in a.ARMS
    assert a.PARAMS["iterations"]<=200
    assert a.PARAMS["allow_writing_files"] is False



def test_scene_exclusion_discards_training_scene_without_using_li(monkeypatch,c,f):
    monkeypatch.setattr(a,"HISTORICAL_N",6)
    frame=c.copy()
    # Arrange identical scene metadata across TRAIN B and held-out TEST A.
    frame.loc[1,"sentinel_scene_id"]=frame.loc[0,"sentinel_scene_id"]
    frame=sample_contract(frame)
    before=list(f["test_indices"])
    train=a.select_train_indices("expanded_exclude_shared_scenes",f,frame,
                                 threshold=f["label_threshold_ppm"])
    assert train.tolist()==[3,5,6]
    assert f["test_indices"]==before
    # Change held-out Li alone: cannot change a scene-based train exclusion.
    frame.loc[0,"Li_icpms"]=1000
    frame=sample_contract(frame)
    again=a.select_train_indices("expanded_exclude_shared_scenes",f,frame,
                                 threshold=f["label_threshold_ppm"])
    assert train.tolist()==again.tolist()
