import numpy as np
import pandas as pd
import pytest
from lithiumscope.experiments.shared.reconstructed_m2_comparison import pair_reference, PREVIOUS_ARM


def fixture():
    n=706
    indices=np.arange(n)
    new=pd.DataFrame({"row_position":indices,"sample_id":["s"+str(i) for i in indices],
        "spatial_group":["g"+str(i//10) for i in indices],"outer_fold":indices//10%5+1,
        "li_ppm":np.linspace(3,100,n),"latitude":-20.,"longitude":-70.,
        "prospectivity_score":np.linspace(.1,.9,n)})
    old=new.rename(columns={"row_position":"test_index","outer_fold":"fold",
                            "prospectivity_score":"score"}).copy()
    old["site_screening_id"]="-20.00000|-70.00000"
    old["arm"]=PREVIOUS_ARM
    return old,new


def test_reference_joins_explicit_ids_and_fixed_cutoff():
    old,new=fixture()
    paired=pair_reference(old,new.iloc[::-1])
    assert len(paired)==706
    assert paired.truth.equals(paired.li_ppm.ge(21.6).astype(int))


@pytest.mark.parametrize("field,value",[("sample_id","unexpected"),("li_ppm",1000),
                                          ("outer_fold",5),("latitude",-21.)])
def test_reconstructed_rejects_mismatched_identity(field,value):
    old,new=fixture()
    new.loc[0,field]=value
    with pytest.raises(ValueError):
        pair_reference(old,new)
