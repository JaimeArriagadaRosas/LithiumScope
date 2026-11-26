"""Meixner identity review must not depend on Li or silently admit a match."""
import numpy as np
import pandas as pd
import pytest
from lithiumscope.experiments.model_1.augmentation.meixner_audit import match_names, raw_view, parse_a1

def query():
    return pd.DataFrame({"excel_row":[41],"source_sample":["QUIS-96h-9"],
        "Latitude":[-23.231833],"Longitude":[-67.232],"SiO2":[66.1],"Li_ppm":[42.]})

def test_identity_independent_of_target_and_ambiguity_retained():
    q=query()
    f=pd.DataFrame({"Sample":["quis96h9","QUIS-96h-9 [123]","QUIS-96h-9"],
        "Latitude":[-23.2318,-23.2318,-23.3],"Longitude":[-67.232]*3,
        "SiO2":[66.2,66.0,66.1],"Li_icpms":[np.nan,900.,42.]},index=[1,2,3])
    edges=match_names(q,f,"test","original.csv")
    assert len(edges)==3
    assert [v["identity_review_candidate"] for v in edges]==[True,True,False]
    assert not any(v["approved_for_training"] for v in edges)
    assert edges==match_names(q.assign(Li_ppm=1000.),f,"test","original.csv")

def test_missing_coordinates_or_silica_only_name_evidence():
    f=pd.DataFrame({"source_sample":["QUIS-96h-9"],"Latitude":[-23.231833],
        "Longitude":[-67.232],"SiO2":[66.1]})
    assert not match_names(query().assign(Latitude=np.nan),f,"test","f")[0]["identity_review_candidate"]
    assert not match_names(query().assign(SiO2=np.nan),f,"test","f")[0]["identity_review_candidate"]

def test_raw_view_keeps_missing_li_and_original_row_position():
    f=pd.DataFrame({"NAME":["q"],"LAT MIN":["-23.2"],"LAT MAX":["-23.0"],
        "LON MIN":["-67.2"],"LI(PPM)":[None],"SIO2(WT%)":["66.1"]},index=[123])
    mapped=raw_view(f,{"NAME":"source_sample","LAT MIN":"Latitude_min","LAT MAX":"Latitude_max",
        "LON MIN":"Longitude_min","LI(PPM)":"Li_icpms","SIO2(WT%)":"SiO2"})
    assert mapped.index.tolist()==[123] and mapped.Li_icpms.isna().all()
    assert mapped.Latitude.iloc[0]==pytest.approx(-23.1)
    assert mapped.SiO2.iloc[0]==pytest.approx(66.1)

def test_unrecognized_a1_layout_rejected():
    with pytest.raises(ValueError,match="layout"):
        parse_a1(pd.DataFrame([[1,2]]))
    with pytest.raises(ValueError,match="header"):
        parse_a1(pd.DataFrame(np.zeros((133,14))))
