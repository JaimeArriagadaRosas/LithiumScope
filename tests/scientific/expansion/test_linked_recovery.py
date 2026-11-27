"""Linked recovery must retain unambiguous identities and reject raw mismatches."""
import pandas as pd
from lithiumscope.experiments.model_1.data_expansion.linked_recovery import one_to_one, corroborate, ANDEAN


def test_one_to_one_rejects_both_directions_of_ambiguity():
    rows=pd.DataFrame({"candidate_id":["a","b","c","d","d","e"],
        "base_row_position":[1,2,2,3,4,5],"candidate_source":[ANDEAN]*6,
        "base_prepared":[True]*6,"base_source":["georoc"]*6,
        "same_sample_name":[True]*6,"chemistry_agreement":[True]*5+[False],
        "shared_chemistry":[12]*6})
    good,bad=one_to_one(rows)
    assert good.candidate_id.tolist()==["a"]
    assert len(bad)==4


def test_raw_link_rechecks_names_and_chemistry_not_only_coordinates():
    frame=pd.DataFrame({"source_sample":["A-1 [123]"],"Latitude":[-30.],
        "Longitude":[-70.],"Li_icpms":[20.],"SiO2":[60.],"TiO2":[1.],
        "Al2O3":[15.],"MgO":[3.],"CaO":[5.]},index=[7])
    old=frame.drop(columns="source_sample").assign(Sample="A1")
    candidate=frame.assign(source_sample="a1")
    assert corroborate(frame,candidate,old).accepted.tolist()==[True]
    assert corroborate(frame.assign(MgO=4.),candidate,old).accepted.tolist()==[False]
    assert corroborate(frame.assign(source_sample="a2"),candidate,old).accepted.tolist()==[False]
    assert corroborate(frame.assign(Li_icpms=21.),candidate,old).accepted.tolist()==[False]
    assert corroborate(frame.drop(columns="MgO"),candidate,old).accepted.tolist()==[False]
