import numpy as np
import pandas as pd

from lithiumscope.experiments.analysis.run_vnext_m1 import (
    harmonize_feot_candidate, heldout_group_masks, IRON_FEOT_TO_FE2O3T,
)


def test_feot_total_conversion_without_double_counting():
    raw=pd.DataFrame([{
        "SAMPLE NAME":"new-1","MATERIAL":"/ WR [12]",
        "LATITUDE MIN":-23.1,"LATITUDE MAX":-23.1,
        "LONGITUDE MIN":-69.1,"LONGITUDE MAX":-69.1,
        "LI(PPM)":24.5,"FEOT(WT%)":6.0,"FE2O3(WT%)":9.0,
        "SIO2(WT%)":60.0,"TIO2(WT%)":1.0,"AL2O3(WT%)":16.0,
        "MNO(WT%)":.1,"MGO(WT%)":4.0,"CAO(WT%)":6.0,
        "NA2O(WT%)":3.0,"K2O(WT%)":3.0,"P2O5(WT%)":.2,
    }])
    copy=raw.copy(deep=True)
    result=harmonize_feot_candidate(raw,"test.csv")
    assert len(result)==1
    assert np.isclose(result.iloc[0]["Fe2O3"],6*IRON_FEOT_TO_FE2O3T)
    assert result.iloc[0]["Fe_total_provenance"]=="FEOT_CONVERTED_TO_FE2O3T"
    assert result.iloc[0]["SUM (no water)"]<103
    pd.testing.assert_frame_equal(raw,copy)


def test_shared_group_heldout_indices_never_overlap():
    base=pd.DataFrame({
        "Longitude":[-70,-70.1,-69,-68,-67,-66,-65],
        "Latitude":[-23,-23.1,-23,-23,-23,-23,-23],
    })
    extra=pd.concat([base, pd.DataFrame({
        "Longitude":[-69.1,-65.1],"Latitude":[-23.1,-23.1],
    })],ignore_index=True)
    splits=heldout_group_masks(base,extra,splits=3)
    assert len(splits)==3
    assert sorted(set(np.concatenate([test for _,test in splits])))==list(range(len(base)))
    assert all(len(train)>0 and len(test)>0 for train,test in splits)
