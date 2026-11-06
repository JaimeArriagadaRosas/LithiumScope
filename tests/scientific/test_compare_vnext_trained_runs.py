"""Comparison safety, matched-sample correctness and spatial uncertainty."""
import numpy as np
import pandas as pd
import pytest

from lithiumscope.experiments.analysis.compare_vnext_trained_runs import (
    check_m2_identical_protocol,
    classification_metrics,
    group_bootstrap_gain,
    pair_shared_m1,
    regression_metrics,
)


def test_m1_common_samples_only_and_ambiguity_excluded():
    old = pd.DataFrame({
        "source_dataset": ["GEOROC"]*4,
        "Longitude": [-70.,-71.,-71.,-72.],
        "Latitude": [-23.,-24.,-24.,-25.],
        "y_true": [3.,4.,4.,5.],
        "spatial_group": ["A","B","B","C"],
        "outer_fold": [1,2,2,3],
        "prediction": [2.,3.,5.,4.],
    })
    new = old.copy()
    new["prediction"] = [3.,3.9,4.1,5.]
    new["outer_fold"] = [2,3,3,1]
    new = pd.concat([new, pd.DataFrame([{
        "source_dataset": "GEOROC", "Longitude": -73.,
        "Latitude": -26., "y_true": 6., "spatial_group": "D",
        "outer_fold": 4, "prediction": 5.5,
    }])], ignore_index=True)
    paired, meta = pair_shared_m1(old,new)
    assert len(paired)==2
    assert meta["excluded_old"]==2
    assert meta["different_outer_fold"]==2
    assert set(paired.y_true)=={3,5}


def test_m1_safety_rejects_wrong_group():
    old = pd.DataFrame([{
        "source_dataset":"GEOROC", "Longitude":-70.,
        "Latitude":-23., "y_true":5., "spatial_group":"A",
        "outer_fold":1, "prediction":4.,
    }])
    new = old.copy()
    new["spatial_group"]="B"
    with pytest.raises(ValueError,match="spatial group"):
        pair_shared_m1(old,new)


def test_paired_m1_rmse_mae_metrics():
    y = np.array([10.,15.,20.])
    o = np.array([8.,10.,22.])
    n = np.array([9.,14.,21.])
    old=regression_metrics(y,o)
    new=regression_metrics(y,n)
    assert old["rmse"]>new["rmse"]
    assert old["mae"]>new["mae"]


def test_group_bootstrap_is_deterministic_and_grouped():
    y=np.array([1.,2.,3.,4.,5.,6.])
    old=np.array([2.,3.,4.,5.,6.,7.])
    new=y.copy()
    group=np.array(["a","a","b","b","c","c"])
    a=group_bootstrap_gain(y,old,new,group,metric="rmse",repeats=100)
    b=group_bootstrap_gain(y,old,new,group,metric="rmse",repeats=100)
    assert a==b
    assert a["spatial_groups"]==3
    assert a["ci95_group_lower"]>0


def test_classification_improves_calibration_without_discrimination_gain():
    y=np.array([0,0,1,1])
    score=np.array([.05,.10,.15,.20])
    transformed=np.array([.35,.40,.60,.65])
    old=classification_metrics(y,score)
    new=classification_metrics(y,transformed)
    assert new["roc_auc"]==old["roc_auc"]
    assert new["average_precision"]==old["average_precision"]
    assert new["brier"]<old["brier"]


def test_m2_checks_frozen_manifest_and_fold_identity(tmp_path):
    old=tmp_path/"old"
    new=tmp_path/"new"
    for directory, sha in ((old,"abc"),(new,"different")):
        (directory/"manifests").mkdir(parents=True)
        (directory/"run.json").write_text(
            '{"state":"completed","summary":{"dataset_sha256":"'+sha+'"}}'
        )
        (directory/"manifests/scientific_folds.json").write_text("[]")
    x=pd.DataFrame({
        "y_true":[0,1], "li_ppm":[2.,3.], "outer_fold":[1,2]
    })
    meta=pd.DataFrame({"spatial_group":["A","B"]})
    with pytest.raises(ValueError,match="fingerprints"):
        check_m2_identical_protocol(old,new,x,x,meta)


def test_bootstrap_refuses_unpaired_inputs():
    with pytest.raises(ValueError,match="differ"):
        group_bootstrap_gain(
            [0,1,0], [.1,.2], [.2,.3,.4],
            ["a","b","c"], metric="brier"
        )


def test_archived_m1_runs_match_by_explicit_observation_keys_if_available():
    from lithiumscope.core.paths import PROJECT_ROOT
    from lithiumscope.experiments.analysis.compare_vnext_trained_runs import (
        OLD_M1, NEW_M1, load_m1_oof,
    )
    directory=PROJECT_ROOT/"results/model_1/runs"
    if not (directory/OLD_M1/"run.json").is_file() or not (
        directory/NEW_M1/"run.json"
    ).is_file():
        pytest.skip("Historical local training artifacts unavailable")
    paired, meta=pair_shared_m1(
        load_m1_oof(directory/OLD_M1),load_m1_oof(directory/NEW_M1)
    )
    assert meta["matched"]>=.8*meta["old_total"]
    assert paired.spatial_group_old.nunique()>5


def test_archived_m2_runs_reuse_identical_spatial_test_protocol_if_available():
    from lithiumscope.core.paths import PROJECT_ROOT
    from lithiumscope.experiments.analysis.compare_vnext_trained_runs import (
        OLD_M2,NEW_M2,load_m2_oof,check_m2_identical_protocol,
    )
    root=PROJECT_ROOT
    old=root/"results/model_2/runs"/OLD_M2
    new=root/"results/model_2/runs"/NEW_M2
    manifest=root/"data/processed/model_2/training_manifest.csv"
    if not old.is_dir() or not new.is_dir() or not manifest.is_file():
        pytest.skip("Historical local training artifacts unavailable")
    a=load_m2_oof(old,"random_forest")
    b=load_m2_oof(new,"random_forest")
    check_m2_identical_protocol(old,new,a,b,pd.read_csv(manifest))
