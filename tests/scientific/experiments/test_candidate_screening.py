import pandas as pd
from lithiumscope.experiments.data_lab.screening.candidates import screen


def test_candidate_screening_flags_cross_source_and_missing_metadata():
    rows=[]
    for i in range(3):
        rows.append({"source_dataset":"candidate","source_file":"part.csv","source_row_position":i,
                     "source_unique_id":str(i),"source_publication":"pub","source_sample":f"s{i}",
                     "Rock_type":"granite","source_tectonic_setting":"CONVERGENT MARGIN",
                     "Latitude":-20.-i,"Longitude":-70.,"Li_icpms":10.,
                     "source_li_raw":10.,"source_li_unit":"ppm","Sample_type":"WHOLE ROCK",
                     "SUM (no water)":100.,"source_li_method":"unknown","source_coordinate_crs":"unknown",
                     **{k:1. for k in ("SiO2","TiO2","Al2O3","Fe2O3","MnO","MgO","CaO","Na2O")}})
    frame=pd.DataFrame(rows)
    base=frame.iloc[:1].copy()
    result=screen(frame,base,{"g_140_220"})
    assert result.screening_state.tolist()==["excluded_screening","convergent_review","convergent_review"]
    assert result.metadata_review_required.all()
    assert not result.approved_for_training.any()
    assert result.existing_spatial_group.tolist()==[True,False,False]
