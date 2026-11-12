"""Manual Model 2 closure: no release, no provisional samples, paired 3-zone QC.

Unit tests do NOT execute the complete user-owned manual training.
"""
from __future__ import annotations
from contextlib import contextmanager
import json
from pathlib import Path
import types

import numpy as np
import pandas as pd
import pytest

from lithiumscope.model_2.training import closure
from lithiumscope.model_2.methodology.three_zone_ranking import ranking_evaluation


def _pairwise_reference():
    rows=[]
    for arm,sign in (("old",-1),("corrected",1)):
        for fold in range(1,6):
            for j in range(9):
                i=(fold-1)*9+j
                li=float(j+2)
                rows.append({
                    "arm":arm,"test_index":i,"fold":fold,
                    "sample_id":f"S{i}",
                    "site_screening_id":f"site{i}",
                    "spatial_group":f"group{fold}",
                    "li_ppm":li,
                    "y_true":int(li>=6),
                    "score":float((j+1)/10 if sign>0 else (9-j)/10),
                })
    return pd.DataFrame(rows)


def test_three_zone_ranking_is_paired_and_never_uses_same_site_twice():
    info,rows=ranking_evaluation(_pairwise_reference(),
                                 triples_per_fold=20,seed=13)
    assert info["candidate_sets_shared_across_arms"] is True
    assert len(rows)==2*5*20
    assert rows.all_three_distinct_sites.all()
    assert info["arms"]["corrected"]["top1_accuracy"]==1.
    assert info["arms"]["old"]["top1_accuracy"]==0.
    assert info["arms"]["corrected"]["folds"]==5
    assert rows.score_tie_abstention.sum()==0


def test_three_zone_ranking_fails_closed_on_mismatched_test_labels():
    data=_pairwise_reference()
    data.loc[(data.arm=="corrected")&(data.sample_id=="S0"),"li_ppm"]=999
    with pytest.raises(ValueError,match="same held-out"):
        ranking_evaluation(data,triples_per_fold=10)


def test_three_zone_ranking_no_site_crosses_outer_folds():
    data=_pairwise_reference()
    data.loc[(data.sample_id=="S10"),"site_screening_id"]="site0"
    with pytest.raises(ValueError,match="multiple held-out folds"):
        ranking_evaluation(data,triples_per_fold=10)


def test_prior_release_params_reuse_exact_estimator_without_optimizing():
    class CatBoostClassifier:
        def get_params(self):
            return {
                "iterations":450,"depth":6,"learning_rate":.04,
                "loss_function":"Logloss","random_seed":12,
                "task_type":"CPU","verbose":True,"allow_writing_files":True
            }
    old={"estimator":CatBoostClassifier()}
    out=closure.prior_release_parameters(old)
    assert out["iterations"]==450
    assert out["depth"]==6
    assert out["random_seed"]==12
    assert out["task_type"]=="CPU"
    assert out["allow_writing_files"] is False
    assert out["thread_count"]==4
    assert closure.LIFECYCLE=="experimental_not_for_individual_release"


def test_model2_menu_option_2_restores_six_algorithm_competition(monkeypatch):
    from lithiumscope.cli import train_command as menu
    from lithiumscope.datasets import provisioner
    from lithiumscope.model_2.training import competition
    called=[]
    monkeypatch.setattr(menu,"choose",lambda *a,**k:"2")
    monkeypatch.setattr(menu,"require_full_training_environment",
                        lambda:called.append("environment"))
    monkeypatch.setattr(menu,"detect_device",lambda **kw:types.SimpleNamespace(
        accelerator="cpu",name="CPU"))
    monkeypatch.setattr(menu,"device_summary",lambda device:None)
    monkeypatch.setattr(menu,"require_model_2_expanded_manifest",
                        lambda:(called.append("expanded_783") or Path("783.csv")))
    @contextmanager
    def fake_log(*args,**kwargs):
        yield "mock log"
    monkeypatch.setattr(menu,"run_log",fake_log)
    monkeypatch.setattr(menu,"register_training_run",lambda *a:None)
    monkeypatch.setattr(competition,"run_model_2_competition",
        lambda manifest,device:(called.append("six_algorithms")
                               or types.SimpleNamespace(
                                   run_dir=Path("results"),winner="catboost",
                                   ranking=pd.DataFrame(),failed=[])))
    menu._run_impl()
    assert called==["environment","expanded_783","six_algorithms"]


def test_model2_menu_option_0_leaves_everything_untouched(monkeypatch):
    from lithiumscope.cli import train_command as menu
    monkeypatch.setattr(menu,"choose",lambda *a,**k:"0")
    monkeypatch.setattr(menu,"require_full_training_environment",
                        lambda:(_ for _ in ()).throw(
                            AssertionError("should never start")))
    assert menu._run_impl() is False


def test_manual_preflight_is_no_training_and_preserves_original_folds(
    monkeypatch,tmp_path
):
    # Test CLI wrapper, without needing to load 783 actual patches.
    values={"state":"ready_for_manual_training","release_authorized":False,
            "GEOROC_provisional_training_rows":0}
    marker=[]
    def fake_prepare(root):
        marker.append("preflight");return (values,)*8
    monkeypatch.setattr(closure,"prepare",fake_prepare)
    output=closure.train(tmp_path,preflight_only=True,output=tmp_path/"out")
    assert output["release_authorized"] is False
    assert marker==["preflight"]
    assert json.loads((tmp_path/"out/preflight.json").read_text())==values
    assert not list((tmp_path/"out").glob("*.joblib"))


def test_smoke_trains_only_bounded_subset_without_persisting_a_model(
    monkeypatch,tmp_path
):
    base=pd.DataFrame({
        "sample_id":[f"S{i}" for i in range(783)],
        "Li_icpms":np.linspace(1.,80.,783),
        "site_screening_id":[f"site{i}" for i in range(783)],
        "spatial_group":[f"G{i//15}" for i in range(783)],
    })
    x=pd.DataFrame(np.ones((783,126)))
    qa=pd.DataFrame({"finite_pixel_fraction":np.ones(783)})
    folds=[{"train_indices":list(range(200)),
            "test_indices":list(range(200,240)),
            "label_threshold_ppm":8.0}]
    monkeypatch.setattr(closure,"prepare",
        lambda root:({"state":"ready"},base,x,folds,qa,
                     np.arange(690),{},pd.DataFrame()))
    monkeypatch.setattr(closure,"define_outer_fold",
        lambda fold,contract:(
            np.arange(200),np.arange(200),np.arange(200,240),8.))
    fit=[]
    def fake_fit(xtr,y,xte,params):
        fit.append((len(xtr),len(xte)))
        return np.full(len(xte),.37)
    monkeypatch.setattr(closure,"_model_from_params",fake_fit)
    dest=tmp_path/"smoke"
    result=closure.train(tmp_path,smoke=True,output=dest)
    assert result["state"]=="smoke_only"
    assert fit==[(125,16)]
    assert result["candidate_created"] is False if "candidate_created" in result else True
    assert not list(tmp_path.rglob("*.joblib"))



def test_candidate_inference_abstains_on_low_quality_image(tmp_path):
    from lithiumscope.model_2.prediction.predictor import score_model_2_image
    image=tmp_path/"unreliable.npy"
    pixels=np.ones((6,32,32),dtype=np.float32)
    pixels[:,:16,:]=np.nan
    np.save(image,pixels,allow_pickle=False)
    bundle={"experimental_only":True,"quality_min_joint_valid":.95,
            "algorithm":"catboost","operating_threshold":.5}
    result,details=score_model_2_image(image,bundle=bundle)
    assert result["prospectivity_score"] is None
    assert result["priority"]=="abstencion"
    assert result["quality_status"]=="ABSTAIN_LOW_SPECTRAL_COVERAGE"
    assert details["joint_valid_fraction"]==.5


def test_three_independent_valid_images_ranked_with_optional_model1_info(
    monkeypatch,tmp_path
):
    from lithiumscope.model_2.methodology.three_zone_ranking import (
        rank_three_candidates
    )
    from lithiumscope.model_2.steps.step_05_spectral_features import (
        extract_spectral_features
    )
    names={}
    for name,red in (("A",1000.),("B",2000.),("C",3000.)):
        p=tmp_path/(name+".npy")
        arr=np.ones((6,32,32),dtype="float32")*2000.
        arr[0]=red
        np.save(p,arr,allow_pickle=False)
        names[name]=p

    class DummyEstimator:
        def predict_proba(self,frame):
            red=float(frame["red_mean"].iloc[0])
            score=float(red/4000.)
            return np.asarray([[1-score,score]])
    columns=list(extract_spectral_features(np.ones((6,32,32))))
    bundle={"estimator":DummyEstimator(),"features":columns,
            "band_names":["red","green","blue","nir","swir16","swir22"],
            "normalize_per_band":False,"algorithm":"catboost",
            "operating_threshold":.5,"quality_min_joint_valid":.95,
            "experimental_only":True}
    result=rank_three_candidates(
        names,bundle=bundle,model1_lithium_ppm={"A":40,"B":20,"C":10})
    assert result["status"]=="RANKED_EXPERIMENTALLY"
    assert result["best_zone"]=="C"
    assert [r["zone"] for r in result["zones"]]==["C","B","A"]
    assert result["zones"][0]["model_1_predicted_li_ppm"]==10.
    assert result["model1_ppm_not_used_to_fit_or_rank"] is True
    image=names["C"]
    arr=np.load(image)
    arr[:,0:20,:]=np.nan
    np.save(image,arr,allow_pickle=False)
    denied=rank_three_candidates(names,bundle=bundle)
    assert denied["status"]=="ABSTAIN_ONE_OR_MORE_ZONES"
    assert denied["best_zone"] is None
    with pytest.raises(ValueError,match="distinct"):
        rank_three_candidates({"A":names["A"],"B":names["A"],"C":names["B"]},
                              bundle=bundle)


def test_release_artifact_not_accepted_as_experimental_ranking(tmp_path):
    from lithiumscope.model_2.methodology.three_zone_ranking import rank_three_candidates
    with pytest.raises(ValueError,match="explicit experimental"):
        rank_three_candidates({"A":tmp_path/"a","B":tmp_path/"b","C":tmp_path/"c"},
                              bundle={})



def test_complete_comparison_writes_all_metrics_without_real_training(
    monkeypatch,tmp_path
):
    """Exercises all output joins and 3-zone evidence on mock fixed predictors."""
    n=783
    li=np.asarray([10.+(i%6)*5 for i in range(n)],dtype=float)
    contract=pd.DataFrame({
        "sample_id":[f"S{i}" for i in range(n)],
        "Li_icpms":li,
        "site_screening_id":[f"site{i}" for i in range(n)],
        "spatial_group":[f"block{i//142}" for i in range(706)]+["provisional"]*77,
    })
    features=pd.DataFrame({"red_mean":li,"green_mean":li*.5})
    qa=pd.DataFrame({"finite_pixel_fraction":[.90 if i%53==0 else 1. for i in range(n)]})
    frozen=[]
    refrows=[]
    for fold in range(1,6):
        te=np.asarray([i for i in range(706) if i//142==fold-1],dtype=int)
        tr=np.setdiff1d(np.arange(706),te)
        t=float(np.quantile(li[tr],.75))
        frozen.append({
            "train_indices":tr.tolist(),
            "test_indices":te.tolist(),
            "label_threshold_ppm":t,
        })
        for j in te:
            for arm in ("historical_706_reference","quality_95_train"):
                refrows.append({
                    "arm":arm,"fold":fold,"test_index":int(j),
                    "sample_id":f"S{j}","site_screening_id":f"site{j}",
                    "spatial_group":f"block{j//142}",
                    "li_ppm":float(li[j]),"fold_li_threshold_ppm":t,
                    "y_true":int(li[j]>=t),
                    "score":float(.2+.012*(li[j]-10)),
                })
    original=pd.DataFrame(refrows)
    fit_calls=[]
    def fake_fit(x,y,te,params):
        fit_calls.append((len(x),len(te),params.get("iterations")))
        z=te["red_mean"].to_numpy(dtype=float)
        return np.clip(.1+(z-10)*.020, .02,.96)
    monkeypatch.setattr(closure,"_model_from_params",fake_fit)
    result=closure._evaluate(tmp_path,tmp_path,contract,features,frozen,
                             qa,{"iterations":669},original)
    assert result["same_706_test_cases"] is True
    assert result["old_release_method_retrained_on_same_folds"]
    assert len(fit_calls)==10
    assert [x[2] for x in fit_calls]==[669,160]*5
    assert len(pd.read_csv(tmp_path/"all_reference_paired_oof.csv"))==706*4
    assert len(pd.read_csv(tmp_path/"per_fold_comparison.csv"))==10
    assert len(pd.read_csv(tmp_path/"three_zone_cases.csv"))>1500
    from lithiumscope.model_2.training.closure_report import write_closure_report
    report=write_closure_report(tmp_path,{"state":"ready"},result)
    assert "passes_descriptive_internal_criteria" in report
    assert (tmp_path/"INFORME_ENTRENAMIENTO_MODELO2_EXPERIMENTAL.md").exists()
    assert (tmp_path/"experimental_acceptance.json").exists()
