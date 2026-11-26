"""Directed transfer must not admit poor spatial precision or source conflicts."""
import numpy as np
import pandas as pd
from lithiumscope.experiments.model_1.augmentation.transfer_preflight import (
    author_year, geographic_ok, static_gate, fold_gate)


def test_static_gate_blocks_inverted_broad_and_quarantined_sources():
    rows=pd.DataFrame({
        "Latitude_min":[-24.,-24.,-24.,-24.,-24.],
        "Latitude_max":[-24.,-25.,-23.,-24.,-24.],
        "Longitude_min":[-67.]*5,"Longitude_max":[-67.]*5,
        "source_publication":["[1]","[1]","[1]","[17558]","[2]"],
        "transfer_review_candidate":[True]*5,"Li_icpms":[20.,40.,45.,60.,55.]})
    refs={"1":{"SMITH A.: Paper [2010]"},"2":{"JONES B.: Paper [2011]"},
          "17558":{"MACDONALD R.: Paper [1992]"}}
    assert geographic_ok(rows).tolist()==[True,False,False,True,True]
    review,aliases,_=static_gate(rows,refs,{"jones:2011"})
    assert review.static_review_candidate.tolist()==[True,False,False,False,False]
    assert aliases[0]=={"smith:2010"}
    changed,_,_=static_gate(rows.assign(Li_icpms=1000.),refs,{"jones:2011"})
    assert changed.static_review_candidate.equals(review.static_review_candidate)


def test_author_year_alias_is_conservative():
    assert author_year("Macdonald et al., 1992")=={"macdonald:1992"}
    assert author_year("MACDONALD R.: A paper [1992] doi: 10.1000/2001")=={"macdonald:1992"}
    assert author_year("[17558]")==set()


def test_fold_gate_blocks_test_groups_publications_and_ood():
    from lithiumscope.experiments.data_lab.candidate_screening import CHEM
    fields=list(CHEM[:9])
    train=pd.DataFrame({c:np.linspace(1.,3.,30) for c in fields})
    q=pd.DataFrame({c:[2.,2.,2.,200.] for c in fields})
    q["candidate_id"]=["safe","group","publication","ood"]
    q["source_dataset"]="chen"
    q["Li_icpms"]=[50.,50.,50.,50.]
    q["spatial_group"]=["g0","g1","g0","g0"]
    q["static_review_candidate"]=True
    flags=fold_gate(q,train,{"g1"},fields,test_aliases={"author:2000"},
                    aliases=[set(),set(),{"author:2000"},set()])
    assert flags.fold_transfer_review_candidate.tolist()==[True,False,False,False]


def test_preflight_summary_counts_only_eligible_tail_rows():
    from lithiumscope.experiments.model_1.augmentation.transfer_preflight import summarize_plans
    part=pd.DataFrame({"fold":[1,1,1,1,1],"source_dataset":["chen"]*4+["puna"],
        "Li_icpms":[40.,41.,70.,20.,25.],
        "static_review_candidate":[True,True,True,False,True],
        "fold_transfer_review_candidate":[True,True,False,False,True]})
    plans,summary=summarize_plans([part,part.assign(fold=2)])
    assert len(plans)==10
    assert plans["tail"].tolist()==[False,True,True,False,False]*2
    assert plans["accepted_tail"].tolist()==[False,True,False,False,False]*2
    assert summary.loc[summary["source_dataset"].eq("chen"),"accepted_tail"].tolist()==[1,1]
    assert summary.loc[summary["source_dataset"].eq("puna"),"accepted_tail"].tolist()==[0,0]
