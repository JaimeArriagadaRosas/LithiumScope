"""Contracts for directed workbook mapping and conservative overlap review."""
import numpy as np
import pandas as pd
import pytest
from lithiumscope.experiments.model_1.augmentation.whole_rock_audit import (
    PUNA, MAC, CHEM, OXIDES, parse_tables, duplicate_edges, summarize, clean, norm_unit)


def workbook():
    samples=pd.DataFrame(np.full((12,16),None,dtype=object))
    data=pd.DataFrame(np.full((14,31),None,dtype=object))
    samples.iat[4,0]="SAMPLE NAME"
    data.iat[6,0]="SAMPLE NAME"
    fields=[*CHEM,"Li_icpms"]
    for j,field in enumerate(fields,7):
        data.iat[3,j]=field.split("_")[0] if field not in OXIDES else ("Fe2O3T" if field=="Fe2O3" else field)
        data.iat[4,j]=1 if field in OXIDES else 3
        data.iat[5,j]="wt%" if field in OXIDES else "mg/kg"
    for i in range(6):
        samples.iloc[6+i,:7]=[f"S{i}",f"ID{i}",-24.,-66.,4200.,"Puna","trachytic_rock"]
        data.iloc[8+i,:3]=[f"S{i}",f"ID{i}","whole rock analyses"]
        data.iloc[8+i,7:30]=[55.,1.,15.,7.,.1,6.,6.,3.,5.,.9]+[10.]*12+[25.+i]
    return samples,data


def test_puna_maps_units_and_exact_sample_identity():
    s,d=workbook()
    frame,inventory,censored=parse_tables(s,d,PUNA)
    assert len(frame)==6 and frame.source_sample.is_unique
    assert frame.Li_icpms.tolist()==[25.,26.,27.,28.,29.,30.]
    assert frame[list(CHEM)].notna().all().all()
    assert frame.Fe2O3.eq(7.).all()
    assert frame.analysis_excel_row.tolist()==list(range(9,15))
    assert not frame.approved_for_training.any()
    assert len(censored)==0
    assert inventory.loc[inventory.canonical_field.eq("Li_icpms"),"unit"].tolist()==["mg/kg"]


def test_incompatible_unit_and_identity_are_rejected():
    s,d=workbook()
    d.iat[5,29]="ng"
    with pytest.raises(ValueError,match="unit"):
        parse_tables(s,d,PUNA)
    s,d=workbook()
    d.iat[8,1]="OTHER"
    with pytest.raises(ValueError,match="IGSN"):
        parse_tables(s,d,PUNA)


def test_duplicate_detection_does_not_depend_on_lithium():
    s,d=workbook()
    q,_,_=parse_tables(s,d,PUNA)
    q=q.iloc[[0]].copy()
    base=q.copy().assign(source_sample="DIFFERENT",source_unique_id="OTHER",Li_icpms=np.nan)
    one=duplicate_edges(q,base,"base")
    two=duplicate_edges(q.assign(Li_icpms=500.),base.assign(Li_icpms=4000.),"base")
    assert one==two and one[0]["possible_duplicate"]
    assert one[0]["shared_chemistry"]==21
    far=base.assign(Latitude=-10.)
    assert not duplicate_edges(q,far,"base")


def test_summary_keeps_macusani_in_publication_quarantine():
    s,d=workbook()
    q,_,_=parse_tables(s,d,PUNA)
    q=q.iloc[:3].copy()
    q.loc[1,"source_li_censoring"]="below"
    q.loc[2,"source_dataset"]=MAC
    q.loc[2,"Li_icpms"]=350.
    review=summarize(q,[],set(),{"q025":4.24,"q975":126.12},set())
    assert review.basic_qc.tolist()==[True,False,True]
    assert review.academic_review_candidate.tolist()==[True,False,False]
    assert review.spatial_review_candidate.tolist()==[True,False,False]
    assert not review.approved_for_training.any()


def test_censoring_and_mass_units():
    assert norm_unit("μg/g")=="ug/g"
    assert norm_unit("µg/g")=="ug/g"
    assert clean("<2")[0]==1.
    assert clean("<2")[1]=="below"
    for value in ("n.a.","ND","-1","inf"):
        assert np.isnan(clean(value)[0])
