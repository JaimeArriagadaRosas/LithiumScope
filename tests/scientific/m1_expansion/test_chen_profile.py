"""Chen profile must distinguish observations from metadata and reject layout drift."""
import numpy as np
import pandas as pd
import pytest
from lithiumscope.experiments.model_1.augmentation.chen_profile import profile


def workbook():
    headers=["CITATIONS","SAMPLE NAME","LATITUDE","LONGITUDE","LOCATION",
             "SIO2(WT%)","TIO2(WT%)","AL2O3(WT%)","FE2O3T(WT%)","MNO(WT%)",
             "MGO(WT%)","CAO(WT%)","NA2O(WT%)","K2O(WT%)","P2O5(WT%)",
             "RB(PPM)","SR(PPM)","LI(PPM)"]
    a=pd.DataFrame(np.full((7,len(headers)),None,dtype=object))
    a.iloc[1]=headers
    for i,li in [(2,50.),(3,"<40"),(4,1000.)]:
        a.iloc[i]=["[1]","sample"+str(i),-24.,-67.,"Andes",55.,1.,15.,7.,.1,6.,6.,3.,5.,.9,50.,100.,li]
    a.iat[5,0]="[1] AUTHOR: PUBLICATION"
    a.iat[6,17]=0
    return a


def test_profile_keeps_exact_positive_li_not_censored_or_references():
    rows,columns,bands,summary=profile(workbook())
    assert rows.Li_icpms.tolist()==[50.,1000.]
    assert rows.excel_row.tolist()==[3,5]
    assert rows.basic_qc.all()
    assert summary["positive_numeric_li_rows"]==2
    assert summary["reference_like_rows_without_chemistry"]==1
    assert summary["academic_tail_basic_qc_rows"]==1
    assert not summary["unique_sample_count_verified"]
    assert not rows.approved_for_training.any()
    assert int(bands.positive_li_rows.sum())==2
    assert columns.loc[columns.excel_column.eq(18),"mapped_field"].item()=="Li_icpms"


def test_profile_does_not_invent_missing_chemistry_or_coordinates():
    a=workbook()
    a.iat[1,2]="unknown latitude header"
    a.iat[1,5]="unknown silica header"
    rows,_,_,summary=profile(a)
    assert not rows.coordinates_valid.any()
    assert not rows.qc_diagnostic.any()
    assert summary["academic_tail_basic_qc_rows"]==0


def test_profile_requires_documented_target_header_and_position():
    a=workbook()
    a.iat[1,17]="LI(PPB)"
    with pytest.raises(ValueError,match="layout"):
        profile(a)


def test_feot_conversion_preserves_both_oxide_bases():
    a=workbook()
    a.iat[1,8]="FEOT(WT%)"
    rows,inventory,_,summary=profile(a)
    assert rows.FeOT_total_as_FeO.eq(7.).all()
    np.testing.assert_allclose(rows.Fe2O3,7.*1.111348)
    np.testing.assert_allclose(rows.oxide_sum-rows.oxide_sum_original_basis,7.*.111348)
    assert rows.basic_qc.all()
    assert summary["original_basis_qc_rows"]==2
    assert inventory.loc[inventory.header.eq("FEOT(WT%)"),"mapped_field"].item()=="FeOT_total_as_FeO"


def test_chen_overlap_is_target_independent_and_uses_location_bounds():
    from lithiumscope.experiments.model_1.augmentation.chen_overlap import overlap_edges
    rows,_,_,_=profile(workbook())
    q=rows.iloc[[0]].copy()
    q["Latitude_min"]=-24.1
    q["Latitude_max"]=-23.9
    base=q.copy().drop(columns=["Latitude_min","Latitude_max"]).assign(Latitude=-23.95,Li_icpms=900.)
    edges=overlap_edges(q,base,[{"georoc:1"}],[{"georoc:2"}],"base")
    assert len(edges)==1 and edges[0]["site_overlap"] and not edges[0]["publication_overlap"]
    assert edges==overlap_edges(q.assign(Li_icpms=1.),base.assign(Li_icpms=2.),
                                [{"georoc:1"}],[{"georoc:2"}],"base")
    far=base.assign(Latitude=-10.)
    assert not overlap_edges(q,far,[{"georoc:1"}],[{"georoc:2"}],"base")
    assert len(overlap_edges(q,far,[{"georoc:1"}],[{"georoc:1"}],"base"))==1
    assert not overlap_edges(q,base.assign(SiO2=80.),[set()],[set()],"base")


def test_bibliography_rejects_ambiguous_and_unresolved_ids():
    from lithiumscope.experiments.model_1.augmentation.chen_overlap import bibliography,publication_sets
    table=pd.DataFrame([["title"],["CITATIONS"],["[1] Author 2010"],["[2] First"],
                        ["[2] Second"],["[1]"]])
    index=bibliography(table)
    resolved,pubs=publication_sets(["[1]","[2]","[3]","[1]; [2]"],index)
    assert resolved.tolist()==[True,False,False,False]
    assert "georoc:1" in pubs[0] and "text:author 2010" in pubs[0]
    assert "georoc:3" in pubs[2]
