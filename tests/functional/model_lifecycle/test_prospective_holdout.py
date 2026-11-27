import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor, DummyClassifier

from lithiumscope.core.hashing import file_sha256
from lithiumscope.model_1.training.preprocessing.tabular import FeatureSchema
from lithiumscope.evaluation.holdout.registry import register, exposure_audit, verified_registration
from lithiumscope.evaluation.holdout.scoring import score_registered
from lithiumscope.evaluation.holdout.evaluation import evaluate_registered


@pytest.fixture
def inputs(tmp_path):
    n=10
    sites=pd.DataFrame({"sample_id":[f"e{i}" for i in range(n)],"source_id":["new_pub"]*n,
                        "site_id":[f"es{i}" for i in range(n)],"spatial_group":[f"eg{i//2}" for i in range(n)],
                        "latitude":np.linspace(-20,-21,n),"longitude":-70.,"region":[f"r{i%3}" for i in range(n)],
                        "SiO2":np.linspace(55,65,n)})
    exposed=pd.DataFrame({"sample_id":["old"],"source_id":["old_pub"],"site_id":["oldsite"],
                          "spatial_group":["oldgroup"],"latitude":[-30.],"longitude":[-70.]})
    labels=pd.DataFrame({"sample_id":sites.sample_id,"source_id":sites.source_id,
                         "li_ppm":np.arange(10,20),"unit":"ppm","analytical_method":"ICP-MS"})
    p,q,r=tmp_path/"sites.csv",tmp_path/"exposed.csv",tmp_path/"labels.csv"
    sites.to_csv(p,index=False)
    exposed.to_csv(q,index=False)
    labels.to_csv(r,index=False)
    review=tmp_path/"review.json"
    review.write_text(json.dumps({"reviewer":"Synthetic reviewer","label_custodian":"Synthetic custodian",
                                  "documentary_basis":"Synthetic regression fixture only",
                                  "complete_exposure_inventory":True,"independent_source_and_site":True,
                                  "labels_unconsulted":True,"site_identity_verified":True,
                                  "coordinates_wgs84_verified":True}),encoding="utf-8")
    models=[]
    for arm,value in (("reference",25.),("candidate",14.5)):
        estimator=DummyRegressor(strategy="constant",constant=value).fit(pd.DataFrame({"SiO2":[55.,65.]}),[11.,18.])
        model=tmp_path/(arm+".joblib")
        joblib.dump({"estimator":estimator,"schema":FeatureSchema(["SiO2"],[],2),
                     "algorithm":"dummy","applicability_profile":{"SiO2":{"q01":0.,"q99":100.}}},model)
        models.append(model)
    return {"root":tmp_path/"registry"/"a","model_group":"model_1","sites_path":p,"exposure_paths":[q],
            "reference":models[0],"candidate":models[1],"review_path":review,
            "label_commitment_sha256":file_sha256(r),"minimum_distance_m":1000.,
            "min_samples":10,"bootstrap_iterations":100},r,sites,exposed


def test_m1_full_blind_registration_scoring_reveal(inputs):
    args,labels,sites,_=inputs
    root=args["root"]
    register(**args)
    assert not (root/"label_reveal.json").exists()
    with pytest.raises(ValueError,match="trusted"):
        score_registered(root)
    scores=score_registered(root,trusted_local_bundles=True)
    assert scores["rows"]==len(sites)
    assert not scores["labels_accessed_by_scoring"]
    with pytest.raises(FileExistsError):
        score_registered(root,trusted_local_bundles=True)
    result=evaluate_registered(root,labels)
    assert result["state"]=="ready_for_review" and result["paired_valid"]==10
    assert result["candidate_metrics"]["rmse"]<result["reference_metrics"]["rmse"]
    assert result["provisional_numeric_gate"]
    assert result["scientific_release_evidence_pass"] is False
    assert result["automatic_model_activation"] is False
    with pytest.raises(FileExistsError):
        evaluate_registered(root,labels)


def test_overlap_source_site_and_distance_block(inputs):
    args,_,sites,exposed=inputs
    for column in ("sample_id","source_id","site_id","spatial_group"):
        modified=sites.copy()
        modified.loc[0,column]=exposed[column].iloc[0]
        with pytest.raises(ValueError,match=column):
            exposure_audit(modified,[exposed],model_group="model_1",minimum_distance_m=1000.)
    modified=sites.copy()
    modified.loc[0,"latitude"]=-30.
    with pytest.raises(ValueError,match="buffer"):
        exposure_audit(modified,[exposed],model_group="model_1",minimum_distance_m=1000.)
    sites["Li_icpms"]=1.
    sites.to_csv(args["sites_path"],index=False)
    with pytest.raises(ValueError,match="labels"):
        register(**args)


def test_tampered_model_blocks_scoring(inputs):
    args,_,_,_=inputs
    register(**args)
    args["candidate"].write_bytes(b"changed")
    with pytest.raises(ValueError,match="changed"):
        score_registered(args["root"],trusted_local_bundles=True)


def test_tampered_scores_or_label_commitment_block_reveal(inputs):
    args,labels,_,_=inputs
    register(**args)
    root=args["root"]
    score_registered(root,trusted_local_bundles=True)
    wrong=labels.with_name("wrong.csv")
    wrong.write_text("uncommitted data",encoding="utf-8")
    with pytest.raises(ValueError,match="commitment"):
        evaluate_registered(root,wrong)
    assert not (root/"label_reveal.json").exists()
    (root/"candidate_scores.csv").write_text("changed",encoding="utf-8")
    with pytest.raises(ValueError,match="changed"):
        evaluate_registered(root,labels)


def test_duplicate_registration_and_changed_site_manifest_block(inputs):
    args,_,_,_=inputs
    register(**args)
    with pytest.raises(ValueError,match="already registered"):
        register(**dict(args,root=args["root"].with_name("b")))
    args["sites_path"].write_text("changed",encoding="utf-8")
    with pytest.raises(ValueError,match="changed"):
        verified_registration(args["root"])


def test_m2_local_images_and_common_endpoint(inputs):
    args,labels,sites,exposed=inputs
    from lithiumscope.model_2.preparation.features.spectral import DEFAULT_BAND_NAMES, extract_spectral_features
    rows=[]
    for i,row in sites.iterrows():
        image=args["sites_path"].parent/f"cube{i}.npy"
        np.save(image,np.full((6,4,4),i+1.,dtype=np.float32),allow_pickle=False)
        meta=image.with_suffix(".json")
        meta.write_text(json.dumps({"scene_id":f"scene{i}","collection":"sentinel-2-l2a",
                                   "latitude":row.latitude,"longitude":row.longitude,"bands":list(DEFAULT_BAND_NAMES)}),
                        encoding="utf-8")
        rows.append(dict(image_path=str(image),image_sha256=file_sha256(image),
                         sidecar_sha256=file_sha256(meta),scene_id=f"scene{i}"))
    sites=pd.concat([sites,pd.DataFrame(rows)],axis=1)
    sites.to_csv(args["sites_path"],index=False)
    exposed["image_sha256"]="a"*64
    exposed["scene_id"]="old_scene"
    exposed.to_csv(args["exposure_paths"][0],index=False)
    features=list(extract_spectral_features(np.ones((6,4,4))))
    for arm,constant in (("reference",0),("candidate",1)):
        estimator=DummyClassifier(strategy="constant",constant=constant).fit(pd.DataFrame(np.zeros((2,len(features)),dtype=float),columns=features),[0,1])
        joblib.dump({"estimator":estimator,"features":features,"algorithm":"dummy","threshold_ppm":15.,
                     "operating_threshold":.5,"feature_extractor_version":3,
                     "applicability_profile":{name:{"q01":-100.,"q99":100.} for name in features}},args[arm])
    args.update(model_group="model_2",label_cutoff_ppm=15.)
    register(**args)
    score_registered(args["root"],trusted_local_bundles=True)
    report=evaluate_registered(args["root"],labels)
    assert report["state"]=="ready_for_review"
    assert report["paired_valid"]==10
    assert report["metrics"]["candidate"]["conventional"]["tp"]==5
    assert report["release_gate"]["scientific_release_evidence_pass"] is False
    assert not report["automatic_model_activation"]
