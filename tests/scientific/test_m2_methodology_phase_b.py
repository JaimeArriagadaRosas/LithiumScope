"""Guardrails for M2 methodology Phase B. Pure synthetic fixtures; no live data."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from lithiumscope.model_2.methodology.contracts import (
    sample_contract, site_summary, policy_counts,
    STATUS_HISTORICAL, STATUS_PROVISIONAL, STATUS_VERIFIED,
)
from lithiumscope.model_2.methodology.spatial import (
    assess_folds, buffered_train_preview,
)
from lithiumscope.model_2.methodology.domain import (
    training_domain_profile, score_domain,
)
from lithiumscope.model_2.methodology.audit import run_audit


@pytest.fixture
def sample_manifest():
    return pd.DataFrame({
        "sample_id": ["A", "B", "C", "D", "E", "F"],
        "image_path": [f"X:\\patch\\{i}.npy" for i in range(6)],
        "Li_icpms": [10., 12., 30., 31., 11., 28.],
        "latitude": [0., .005, .1, .2, .3, .4],
        "longitude": [-70., -70., -70., -70., -70., -70.],
        "spatial_group": [f"g{i}" for i in range(6)],
        "sentinel_scene_id":["scene1","scene1","scene2","scene3","scene4","scene5"],
        "source_citation":["p1","p1","p2","p3","p4","p5"],
        "coordinate_verification":["FIELD_VERIFIED_SURFACE",
            "PRE_SCREENED_NOT_FIELD_VERIFIED", "", "", "", ""],
        "source_dataset":["FIELD","GEOROC_B_SURFACE_PROVISIONAL",
                          "OLD","OLD","OLD","OLD"],
    })


@pytest.fixture
def two_folds():
    return [
        {"train_indices":[1,3,5], "test_indices":[0,2,4]},
        {"train_indices":[0,2,4], "test_indices":[1,3,5]},
    ]


def test_contract_does_not_modify_source(sample_manifest):
    before=sample_manifest.copy(deep=True)
    audited=sample_contract(sample_manifest)
    assert_frame_equal(sample_manifest,before)
    assert audited.sample_id.tolist()==list("ABCDEF")
    assert len(audited)==6
    assert audited.patch_path_id.nunique()==6


def test_missing_and_duplicate_ids_rejected(sample_manifest):
    with pytest.raises(ValueError,match="provenance"):
        sample_contract(sample_manifest.drop(columns=["image_path"]))
    dup=sample_manifest.copy()
    dup.loc[1,"sample_id"]="A"
    with pytest.raises(ValueError,match="duplicate"):
        sample_contract(dup)


@pytest.mark.parametrize("field,value",[
    ("latitude",91),("longitude",-181),("Li_icpms",-1),
    ("latitude",np.nan),("Li_icpms",np.inf),
])
def test_invalid_geological_inputs_rejected(sample_manifest,field,value):
    frame=sample_manifest.copy()
    frame.loc[0,field]=value
    with pytest.raises(ValueError):
        sample_contract(frame)


def test_eligibility_never_inferrs_field_verification(sample_manifest):
    frame=sample_contract(sample_manifest)
    assert frame.label_eligibility.tolist()==[
        STATUS_VERIFIED,STATUS_PROVISIONAL,
        STATUS_HISTORICAL,STATUS_HISTORICAL,
        STATUS_HISTORICAL,STATUS_HISTORICAL
    ]
    assert policy_counts(frame)["verified_surface_samples"]==1
    assert policy_counts(frame)["eligibility"][STATUS_HISTORICAL]==4


def test_conflict_detected_but_not_forced_to_one_label(sample_manifest):
    frame=sample_manifest.copy()
    frame.loc[1,["latitude","longitude"]]=frame.loc[0,["latitude","longitude"]]
    frame.loc[1,"Li_icpms"]=42.
    result=sample_contract(frame)
    assert result.loc[:1,"site_has_conflicting_labels"].tolist()==[True,True]
    assert result.loc[0,"high_li_descriptive"]!=result.loc[1,"high_li_descriptive"]
    assert len(site_summary(result))==5
    assert policy_counts(result)["conflicting_sites"]==1


def test_groups_are_isolated_but_scene_dependence_is_reported(sample_manifest,two_folds):
    contract=sample_contract(sample_manifest)
    original=json.loads(json.dumps(two_folds))
    summary,rows=assess_folds(contract,two_folds,warning_distance_m=1000)
    assert two_folds==original
    assert len(rows)==6
    assert summary.n_test.sum()==6
    assert summary.n_test_scene_shared.sum()==2
    assert summary.n_test_too_close.sum()==2
    assert rows.test_index.nunique()==6


def test_unknown_scene_not_spurious_dependency(sample_manifest,two_folds):
    frame=sample_manifest.copy()
    frame["sentinel_scene_id"]=""
    summary,_=assess_folds(sample_contract(frame),two_folds)
    assert summary.n_test_scene_shared.sum()==0


def test_group_leakage_rejected(sample_manifest,two_folds):
    frame=sample_manifest.copy()
    frame.loc[1,"spatial_group"]="g0"
    with pytest.raises(ValueError,match="spatial groups"):
        assess_folds(sample_contract(frame),two_folds)


def test_site_leakage_rejected_even_when_groups_differ(sample_manifest,two_folds):
    frame=sample_manifest.copy()
    frame.loc[1,["latitude","longitude"]]=frame.loc[0,["latitude","longitude"]]
    with pytest.raises(ValueError,match="sites"):
        assess_folds(sample_contract(frame),two_folds)


def test_repeated_and_malformed_fold_rejected(sample_manifest,two_folds):
    contract=sample_contract(sample_manifest)
    repeat=[{"train_indices":[1,3,5],"test_indices":[0,2,4]}]*2
    with pytest.raises(ValueError,match="exactly one"):
        assess_folds(contract,repeat)
    bad=[{"train_indices":[0,3,5],"test_indices":[0,2,4]}]
    with pytest.raises(ValueError,match="overlap"):
        assess_folds(contract,bad)


def test_buffer_preview_only_removes_train_and_preserves_test(sample_manifest,two_folds):
    c=sample_contract(sample_manifest)
    b=buffered_train_preview(c,two_folds,buffer_m=1000)
    assert b.n_removed_buffer.tolist()==[1,1]
    assert b.test_indices_unchanged.tolist()==[(0,2,4),(1,3,5)]
    assert b.n_train_remaining.tolist()==[2,2]
    assert b.train_indices_remaining.tolist()==[(3,5),(2,4)]
    assert buffered_train_preview(c,two_folds,buffer_m=0).n_removed_buffer.sum()==0


def test_scene_filter_is_explicit_and_reduces_train_only(sample_manifest,two_folds):
    c=sample_contract(sample_manifest)
    buffered=buffered_train_preview(c,two_folds,buffer_m=0)
    with_scenes=buffered_train_preview(c,two_folds,buffer_m=0,exclude_shared_scenes=True)
    assert with_scenes.n_train_remaining.sum() < buffered.n_train_remaining.sum()
    assert with_scenes.test_indices_unchanged.tolist()==buffered.test_indices_unchanged.tolist()


def test_train_domain_profile_frozen_against_test(sample_manifest):
    x=pd.DataFrame({"band":[1.,2.,3.,4.],"index":[5.,6.,7.,8.]})
    original=x.copy(deep=True)
    profile=training_domain_profile(x.iloc[:3])
    distant=pd.DataFrame({"band":[999.], "index":[900.]})
    score=score_domain(distant,profile)
    assert score.ood_fraction.iloc[0]==1
    assert score.ood_alert.iloc[0]
    assert profile["count"]==3
    assert profile["source"]=="train_only"
    assert_frame_equal(x,original)
    assert training_domain_profile(x.iloc[:3])==profile


def test_domain_rejects_unordered_missing_nonfinite_train_features():
    x=pd.DataFrame({"a":[0.,1.,2.], "b":[2.,3.,4.]})
    profile=training_domain_profile(x)
    with pytest.raises(ValueError,match="order"):
        score_domain(x[["b","a"]],profile)
    with pytest.raises(ValueError,match="order"):
        score_domain(x[["a"]],profile)
    with pytest.raises(ValueError,match="finite"):
        training_domain_profile(pd.DataFrame({"x":[1.,np.nan]}))


def test_cli_generates_isolated_artifacts_and_has_nonzero_strict_gate(
    tmp_path,sample_manifest,two_folds
):
    manifest=tmp_path/"manifest.csv"
    features=tmp_path/"features.csv"
    fold_path=tmp_path/"folds.json"
    sample_manifest.to_csv(manifest,index=False)
    pd.DataFrame({
        "band_mean":[1.,2.,3.,4.,5.,6.],
        "ratio_mean":[2.,3.,4.,5.,6.,7.],
        "Li_icpms":sample_manifest.Li_icpms,
        "spatial_group":sample_manifest.spatial_group,
    }).to_csv(features,index=False)
    fold_path.write_text(json.dumps(two_folds),encoding="utf-8")
    out=tmp_path/"audit"
    result=run_audit(manifest,features,fold_path,out,buffer_m=1000,strict=True)
    assert result["model_fitted"] is False
    assert result["original_folds_modified"] is False
    assert result["strict_policy_pass"] is False
    assert result["folds"]==2
    assert result["test_rows_sharing_scene"]==2
    assert (out/"phase_b_preflight.json").exists()
    assert (out/"fold_train_only_domain_profiles.json").exists()
    assert json.loads(fold_path.read_text(encoding="utf-8"))==two_folds
    assert len(pd.read_csv(out/"fold_train_only_domain_scores.csv"))==6


def test_cli_refuses_reordered_feature_cache(tmp_path,sample_manifest,two_folds):
    man=tmp_path/"manifest.csv";feat=tmp_path/"features.csv";f=tmp_path/"f.json"
    sample_manifest.to_csv(man,index=False)
    reordered=sample_manifest.iloc[::-1].reset_index(drop=True)
    pd.DataFrame({"x":np.arange(6.),"Li_icpms":reordered.Li_icpms,
                  "spatial_group":reordered.spatial_group}).to_csv(feat,index=False)
    f.write_text(json.dumps(two_folds),encoding="utf-8")
    with pytest.raises(ValueError,match="align"):
        run_audit(man,feat,f,tmp_path/"output")


from lithiumscope.model_2.methodology.decision import (
    freeze_inner_oof_threshold, evaluate_frozen_threshold,
)


@pytest.fixture
def inner_oof_example():
    return pd.DataFrame({
        "row_index":list(range(8)),
        "group":list("aabbccdd"),
        "inner_fold":[0,0,0,0,1,1,1,1],
        "y_true":[0,1,0,1,0,1,0,1],
        "score":[.12,.74,.18,.71,.15,.80,.19,.79],
    })


def test_inner_oof_threshold_only_reads_outer_train(inner_oof_example):
    allowed=list(range(8))
    reserved=[8,9]
    before=inner_oof_example.copy(deep=True)
    frozen=freeze_inner_oof_threshold(inner_oof_example,
        permitted_outer_train_indices=allowed,
        reserved_outer_test_indices=reserved)
    assert frozen.provenance=="inner_group_oof_train_only"
    assert frozen.inner_count==8 and frozen.inner_folds==2
    assert_frame_equal(inner_oof_example,before)
    # The frozen policy cannot change when outer labels change.
    t=frozen.selection.threshold
    a=evaluate_frozen_threshold([0,1],[.1,.7],frozen)
    b=evaluate_frozen_threshold([1,0],[.1,.7],frozen)
    assert frozen.selection.threshold==t
    assert a["recall"]!=b["recall"]


def test_inner_oof_refuses_outer_data_or_missing_train_rows(inner_oof_example):
    frame=inner_oof_example.copy()
    frame.loc[0,"row_index"]=8
    with pytest.raises(ValueError,match="ONLY"):
        freeze_inner_oof_threshold(frame,
            permitted_outer_train_indices=list(range(8)),
            reserved_outer_test_indices=[8,9])
    with pytest.raises(ValueError,match="ONLY"):
        freeze_inner_oof_threshold(inner_oof_example.iloc[:-1],
            permitted_outer_train_indices=list(range(8)),
            reserved_outer_test_indices=[8,9])


def test_inner_oof_refuses_groups_split_across_inner_folds(inner_oof_example):
    frame=inner_oof_example.copy()
    frame.loc[1,"inner_fold"]=1
    with pytest.raises(ValueError,match="fragmented"):
        freeze_inner_oof_threshold(frame,
            permitted_outer_train_indices=list(range(8)),
            reserved_outer_test_indices=[8,9])


def test_inner_oof_rejects_invalid_scores_and_one_class_folds(inner_oof_example):
    wrong=inner_oof_example.copy()
    wrong.loc[0,"score"]=3.0
    with pytest.raises(ValueError,match="Scores"):
        freeze_inner_oof_threshold(wrong,
            permitted_outer_train_indices=list(range(8)),
            reserved_outer_test_indices=[8,9])
    wrong=inner_oof_example.copy()
    wrong.loc[wrong.inner_fold==1,"y_true"]=0
    with pytest.raises(ValueError,match="both labels"):
        freeze_inner_oof_threshold(wrong,
            permitted_outer_train_indices=list(range(8)),
            reserved_outer_test_indices=[8,9])
