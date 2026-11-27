"""Feature inventory must never turn target aliases or metadata into predictors."""
import numpy as np
import pandas as pd

from lithiumscope.experiments.model_1.data_expansion.feature_audit import classify, derived, numeric_measurement, profile


def test_explicit_element_allowlist_excludes_target_and_identifiers():
    assert classify("Rb_icpms",["Rb_icpms"])=="existing"
    for col in ("La_icpms","Ce_xrf","Ni (ppm)","Be_ppb"):
        assert classify(col,[])=="explicit_element_measurement_review"
    for col in ("Li_icpms","Li (ppm)","Lithium_ppm","Li_xrf","Li","source_li_raw",
                "sample_id","Geochemistry Reference","prediction","SUM (no water)","Li_Rb_ratio"):
        assert classify(col,[])=="not_allowlisted"
    assert classify("La",[])=="element_unit_unknown_review"
    assert classify("Age (Ma)",[])=="age_review"


def test_derived_chemistry_rejects_nonpositive_ratio_inputs_and_uses_molar_units():
    frame=pd.DataFrame({"Rb_icpms":[10.,0.,-1.],"Cs_icpms":[2.,1.,1.],
                        "Al2O3":[101.961]*3,"CaO":[56.077,0.,np.nan],
                        "Na2O":[61.979,0.,0.],"K2O":[94.196,0.,0.]})
    out=derived(frame)
    assert np.isclose(out["log_ratio_Rb_icpms_Cs_icpms"].iloc[0],np.log(5.))
    assert out["log_ratio_Rb_icpms_Cs_icpms"].iloc[1:].isna().all()
    assert np.isclose(out.molar_A_CNK.iloc[0],1./3.)
    assert out.molar_A_CNK.iloc[1:].isna().all()
    assert out.index.equals(frame.index)


def test_numeric_profiles_preserve_missingness_and_groups():
    x=numeric_measurement(pd.Series(["5","invalid",-1.,np.inf,0.]))
    assert x.notna().tolist()==[True,False,False,False,True]
    y=pd.Series([5.,6.,7.,8.,9.])
    g=pd.Series(["a","a","b","b","c"])
    p=profile(x,y,g)
    assert p["observed"]==2 and p["groups_observed"]==2
    assert p["coverage"]==.4 and p["zero"]==1
    assert np.isnan(p["spearman_descriptive"])
