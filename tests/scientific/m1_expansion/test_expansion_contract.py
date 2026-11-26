"""Scientific contracts for the real-data M1 expansion, without fitting models."""
from types import SimpleNamespace

import pandas as pd
import pytest

from lithiumscope.core.hashing import file_sha256
from lithiumscope.experiments.data_lab.candidate_screening import CHEM
from lithiumscope.experiments.model_1.expansion.identity import resolve
from lithiumscope.experiments.model_1.expansion.inputs import _raw_metadata
from lithiumscope.experiments.model_1.expansion.cohorts import (
    review_candidates, reserve_holdout, fold_membership)


def candidate(source, file, row, **overrides):
    return {"source_dataset":source,"source_file":file,"source_row_position":row,
            "source_unique_id":"id1","source_publication":"pub1","source_sample":"A",
            "Rock_type":"granite","source_tectonic_setting":"convergent margin",
            "Latitude":-20.1,"Longitude":-69.1,"Li_icpms":25.,
            "SUM (no water)":99.,"Sample_type":"whole rock","source_li_unit":"ppm",
            "source_li_raw":"25","source_li_method":"unknown","source_coordinate_crs":"unknown",
            **{key:1. for key in CHEM},**overrides}


def test_identical_copies_can_be_recovered_but_historical_overlap_cannot():
    a=candidate("georoc_andean_202609","a.csv",0)
    b=candidate("georoc_felsic_rocks","b.csv",1)
    rows=pd.DataFrame([a,b])
    baseline=pd.DataFrame([candidate("old","old.csv",0,source_unique_id="other",
         source_publication="other",source_sample="Z",Latitude=-40.,Longitude=-72.)])
    audit=resolve(rows,baseline,[])
    assert audit.exact_copy.tolist()==[False,True]
    assert audit.screening_state.tolist()==["convergent_review","exact_candidate_copy"]
    assert audit.canonical_id.nunique()==1
    baseline=pd.DataFrame([a])
    blocked=resolve(rows,baseline,[])
    assert blocked.screening_state.tolist()==["excluded_screening","exact_candidate_copy"]


def test_conflicting_measurements_are_not_canonical_copies():
    a=candidate("georoc_andean_202609","a.csv",0)
    b=candidate("georoc_felsic_rocks","b.csv",1,Li_icpms=30.,source_li_raw="30")
    audit=resolve(pd.DataFrame([a,b]),pd.DataFrame([a]).iloc[:0],[])
    assert not audit.exact_copy.any()
    assert audit.screening_state.eq("excluded_screening").all()


def test_raw_metadata_only_from_explicit_verified_columns(tmp_path):
    spec=SimpleNamespace(key="source")
    path=tmp_path/"source"/"a.csv"
    path.parent.mkdir()
    path.write_text("LI(PPM),LI METHOD,COORDINATE CRS\n25,ICP-MS,EPSG:4326\n",encoding="utf-8")
    frame=pd.DataFrame([candidate("source","a.csv",0)])
    m={"source_sha256":{"a.csv":file_sha256(path)}}
    recovered,inventory=_raw_metadata(spec,frame,m,tmp_path)
    assert recovered.source_li_method.iloc[0]=="ICP-MS"
    assert recovered.source_coordinate_crs.iloc[0]=="EPSG:4326"
    assert inventory[0]["raw_state"]=="verified"
    path.write_text("LI(PPM),METHOD\n25,unknown\n",encoding="utf-8")
    with pytest.raises(ValueError,match="differs"):
        _raw_metadata(spec,frame,m,tmp_path)
    m={"source_sha256":{"a.csv":file_sha256(path)}}
    unchanged,_=_raw_metadata(spec,frame,m,tmp_path)
    assert unchanged.source_li_method.iloc[0]=="unknown"


def review_frame():
    return pd.DataFrame({"candidate_id":["a","b","c","d"],"raw_source_sha256":["h"]*4,
        "screening_state":["convergent_review","high_transfer_review","other_domain_review","excluded_screening"],
        "raw_verified":[True]*4,"prepared_coordinates_valid":[True]*4,
        "spatial_group":["g1","new","new2","new3"],"source_publication":["other","p2","p1","p4"]})


def ledger(identifier, decision, digest="h"):
    return {"candidate_id":identifier,"raw_source_sha256":digest,"decision":decision,
            "method_evidence":"ICP-MS, publication p. 4","crs_evidence":"EPSG:4326, appendix",
            "reference":"source record","notes":"experiment only"}


def test_no_automatic_approval_and_review_requires_evidence_and_identity():
    data=review_frame()
    assert not review_candidates(data).experimental_eligible.any()
    for row in (ledger("a","compatible","wrong"),ledger("d","compatible"),ledger("b","compatible"),
                {**ledger("a","compatible"),"method_evidence":""}):
        with pytest.raises(ValueError):
            review_candidates(data,pd.DataFrame([row]))
    reviewed=review_candidates(data,pd.DataFrame([ledger("a","compatible"),ledger("b","experimental_transfer")]))
    assert reviewed.experimental_eligible.tolist()==[True,True,False,False]
    assert not reviewed.approved_for_production.any()


def test_fold_plan_never_admits_shared_test_groups_or_publications():
    data=review_frame()
    reviewed=review_candidates(data,pd.DataFrame([ledger("a","compatible"),
        ledger("b","experimental_transfer"),ledger("c","compatible")]))
    reviewed=reserve_holdout(reviewed,{"g1","g2"},{"p1","p2"})
    x=pd.DataFrame({"SiO2":[10.,11.,10.,11.]})
    base=pd.DataFrame({"SiO2":[10.,11.,10.,11.]})
    folds=pd.DataFrame({"outer_fold":[1,1,2,2],"spatial_group":["g1","g1","g2","g2"]})
    membership=fold_membership(reviewed,x,base,folds,pd.Series(["p1","p1","p2","p2"]),["SiO2"])
    first=membership.loc[membership.fold.eq(1)&membership.cohort.eq("D4")]
    assert first.set_index("candidate_id").candidate_train.to_dict()=={"a":False,"b":True,"c":False,"d":False}
    second=membership.loc[membership.fold.eq(2)&membership.cohort.eq("D4")]
    assert not second.set_index("candidate_id").loc["b","candidate_train"]


def test_reserved_groups_and_publications_never_enter_pool():
    data=pd.DataFrame({"experimental_eligible":[True]*6,
         "spatial_group":[f"new{i}" for i in range(5)]+["other"],
         "source_publication":["shared"]*6})
    reserved=reserve_holdout(data,set(),set())
    assert reserved.candidate_partition.eq("reserved_development_check").sum()>=1
    assert not reserved.candidate_partition.eq("train_pool").any()
