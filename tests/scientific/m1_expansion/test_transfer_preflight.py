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
