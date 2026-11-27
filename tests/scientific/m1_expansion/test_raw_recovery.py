"""Raw recovery requires independent chemical confirmation and unambiguous lineage."""
import numpy as np
import pandas as pd

from lithiumscope.experiments.data_lab.source_studies.geochemistry import measurement_map, numeric_ppm
from lithiumscope.experiments.model_1.data_expansion.raw_recovery import unique_matches


def test_only_explicit_additional_element_units_are_recovered():
    assert measurement_map(["LA(PPM)","Y (PPB)","LI(PPM)","RB(PPM)","Age","SAMPLE",
                            "Ce(PPM)","CE(PPB)"])=={"La":("LA(PPM)","ppm"),"Y":("Y (PPB)","ppb")}


def test_units_and_censoring():
    result=numeric_ppm(pd.Series(["1000","<200","ND","-5","bad"]),"ppb")
    assert np.allclose(result.iloc[:2],[1.,.1])
    assert result.iloc[2:].isna().all()


def samples():
    return pd.DataFrame({"source_sample":[" ABC "],"Latitude":[-30.],"Longitude":[-70.],
        "Li_icpms":[30.],"SiO2":[60.],"TiO2":[1.],"Al2O3":[15.],"MgO":[3.],"CaO":[5.]},index=[10])


def test_unique_match_requires_agreement_and_rejects_ambiguous_rows():
    base=samples()
    raw=base.copy().set_axis([7]).assign(source_sample="abc")
    match=unique_matches(base,raw)
    assert match.state.tolist()==["matched"] and match.raw_position.tolist()==[7]
    assert unique_matches(base,raw.assign(MgO=4.)).state.tolist()==["core_or_chemistry_mismatch"]
    assert unique_matches(base,raw.drop(columns="MgO")).state.tolist()==["core_or_chemistry_mismatch"]
    ambiguous=pd.concat([raw,raw.set_axis([8])])
    assert unique_matches(base,ambiguous).state.tolist()==["ambiguous"]
    repeated=pd.concat([base,base.set_axis([11])])
    assert unique_matches(repeated,raw).state.tolist()==["raw_row_reused","raw_row_reused"]
