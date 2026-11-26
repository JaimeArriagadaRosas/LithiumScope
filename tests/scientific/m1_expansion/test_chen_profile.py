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
