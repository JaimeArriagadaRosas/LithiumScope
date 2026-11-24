import numpy as np
import pandas as pd
import pytest

from lithiumscope.model_1.preparation.source_reconciliation import (
    CHEMISTRY, FIELDS, reconcile_georoc, reported_rock_label, oxide_balance,
)
from lithiumscope.model_1.training.preprocessing.tabular import OXIDES


def pair():
    common={c:1. for c in CHEMISTRY}
    common.update(source_sample="sample",Li_icpms=10.,Latitude=-20.,Longitude=-70.,
                  Fe_total_provenance="ORIGINAL_FE2O3")
    historical=pd.DataFrame([common],index=[17])
    source=pd.DataFrame([{**common,**dict.fromkeys(FIELDS,"verified"),"source_unique_id":"001"},
                         {**common,**dict.fromkeys(FIELDS,"verified"),"source_unique_id":"002",
                          "Fe2O3":1.01}])
    return historical,source


def test_chemistry_disambiguates_identity_candidates():
    historic,source=pair()
    result=reconcile_georoc(historic,source).iloc[0]
    assert result.source_row==17
    assert result.identity_candidates==2
    assert result.chemical_matches==1
    assert result.source_unique_id=="001"
    assert result.state=="unique"


def test_equal_chemistry_does_not_break_ties():
    historic,source=pair()
    source.loc[1,"Fe2O3"]=1.
    result=reconcile_georoc(historic,source).iloc[0]
    assert result.state=="ambiguous"
    assert result.chemical_matches==2
    assert "source_unique_id" not in result.index or pd.isna(result.source_unique_id)


def test_missing_and_zero_differ_and_fe_conventions_must_agree():
    historic,source=pair()
    historic.loc[17,"U_icpms"]=np.nan
    source["U_icpms"]=0.
    assert reconcile_georoc(historic,source).iloc[0].state=="chemistry_mismatch"
    source.loc[0,"U_icpms"]=np.nan
    source.loc[0,"Fe_total_provenance"]="FEOT_CONVERTED_TO_FE2O3T"
    assert reconcile_georoc(historic,source).iloc[0].state=="chemistry_mismatch"
    source.loc[0,"Fe_total_provenance"]="ORIGINAL_FE2O3"
    assert reconcile_georoc(historic,source).iloc[0].state=="unique"


def test_missing_keys_never_match():
    historic,source=pair()
    historic.loc[17,"Longitude"]=np.nan
    source["Longitude"]=np.nan
    result=reconcile_georoc(historic,source).iloc[0]
    assert result.state=="missing_identity_key"
    assert result.chemical_matches==0


@pytest.mark.parametrize("raw,expected",[
    ("/ TUFF [30009]","tuff"),
    ("/ BASALTIC-ANDESITE [123] / NOT GIVEN [345]","basaltic-andesite"),
    ("/ ANDESITE [12] / ANDESITE [34]","andesite"),
    ("/ ANDESITE [12] / BASALT [34]",None),
    ("/ NOT GIVEN [12]",None),
    (None,None),
])
def test_lithology_only_restores_unambiguous_reported_name(raw,expected):
    assert reported_rock_label(raw)==expected


def test_balance_comparison_respects_separate_ferrous_oxide():
    frame=pd.DataFrame([{**dict.fromkeys(OXIDES,9.9),"FeO":1.,"SUM (no water)":100.},
                        {**dict.fromkeys(OXIDES,10.),"FeO":np.nan,"SUM (no water)":100.}])
    balance=oxide_balance(frame)
    assert balance.matches_10_within_0_05.tolist()==[False,True]
    assert balance.matches_10_plus_feo_within_0_05.tolist()==[True,True]
    assert balance.reported_feo.isna().tolist()==[False,True]
