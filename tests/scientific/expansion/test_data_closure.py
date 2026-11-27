"""Closing data-composition policies preserve the historical population and schema."""
import numpy as np
import pandas as pd
import pytest
from lithiumscope.experiments.model_1.studies.augmentation.data_closure import transfer_weights, candidate_features


def test_transfer_cap_per_band_and_tail_boundary():
    historical=pd.Series([8.]*10+[18.]*20+[40.]*10+[45.]*4+[80.]*2)
    candidates=pd.Series([8.]*20+[18.]*2+[40.]*10+[41.,45.,46.,55.,90.],
                         index=[f"c{i}" for i in range(37)])
    w=transfer_weights(historical,candidates)
    assert w.index.equals(candidates.index)
    assert np.isclose(w[candidates.eq(8)].sum(),1.)
    assert np.isclose(w[candidates.eq(18)].sum(),.2)
    assert np.isclose(w[candidates.eq(40)].sum(),1.)
    assert np.isclose(w[candidates.between(41,55)].sum(),.4)
    assert w[candidates.eq(90)].item()==0.
    tail=transfer_weights(historical,candidates,tail_only=True)
    assert tail[candidates.le(40)].eq(0).all()
    assert tail[candidates.gt(40)].equals(w[candidates.gt(40)])
    assert (w[candidates.le(40)]<=.1).all() and (w<=.25).all()


def test_transfer_never_invents_an_unrepresented_band_or_invalid_target():
    h=pd.Series([12.,20.,40.])
    assert transfer_weights(h,pd.Series([50.,120.])).eq(0.).all()
    with pytest.raises(ValueError):
        transfer_weights(h,pd.Series([0.,12.]))


def test_candidate_features_recompute_ratios_and_keep_missing_methods():
    index=pd.Index(["chen:1"],name="candidate_id")
    rows=pd.DataFrame({"SiO2":[70.],"Fe2O3":[2.],"MgO":[1.],"Na2O":[3.],"K2O":[4.],
        "Al2O3":[15.],"CaO":[2.],"Li_icpms":[50.],"Rb_icpms":[80.],"Sr_icpms":[40.],
        "georoc_La_ppm":[20.],"georoc_Yb_ppm":[2.],"La_icpms":[999.]},index=index)
    x=rows[["SiO2","Fe2O3","MgO","Na2O","K2O","Al2O3","CaO","Rb_icpms","Sr_icpms"]].copy()
    for col in ("Rock_type","Alkali_Sum","Mg_Number","A_CNK_proxy","K_Mg_ratio"):
        x[col]=np.nan
    measured=pd.DataFrame(columns=["La_icpms","georoc_La_ppm","georoc_Yb_ppm"])
    result=candidate_features(rows,x,measured)
    assert "Li_icpms" not in result
    assert result.Rock_type.isna().all() and result.La_icpms.isna().all()
    assert result.Alkali_Sum.item()==7.
    assert np.isclose(result.log_ratio_Rb_icpms_Sr_icpms.item(),np.log(2.))
    assert np.isclose(result.log_ratio_georoc_La_ppm_georoc_Yb_ppm.item(),np.log(10.))
