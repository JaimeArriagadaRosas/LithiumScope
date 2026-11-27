import json
import pandas as pd
import pytest

from lithiumscope.datasets.adapters.georoc.adapter import GeorocAdapterConfig, harmonize_georoc_frame
from lithiumscope.datasets.adapters.georoc.ingestion import ingest_georoc_files

def test_conflicting_aliases_stop_ingestion(tmp_path):
    source = tmp_path/"source.csv"
    source.write_text("MATERIAL,LI(PPM),LI,LONGITUDE,LATITUDE\nWR,10,99,-70,-30\n")
    with pytest.raises(ValueError, match="Conflicting GEOROC aliases"):
        ingest_georoc_files([source],tmp_path/"out.csv",minimum_predictors=0)
    assert not (tmp_path/"out.csv").exists()

def test_coordinate_bounds_and_streaming_censoring(tmp_path):
    source=tmp_path/"source.csv"
    source.write_text("MATERIAL,LI(PPM),LONGITUDE MIN,LONGITUDE MAX,LATITUDE MIN,LATITUDE MAX\nWR,<10,-71,-69,-31,-29\nWR,15,-69,-71,-31,-29\n")
    records=[]
    result=ingest_georoc_files([source],tmp_path/"out.csv",minimum_predictors=0,
                              censoring_callback=records.extend,chunksize=1)
    assert result.rows_kept==1
    audit=json.loads(result.audit_path.read_text())
    assert audit["rejection_counts"]=={"inconsistent_coordinate_bounds":1}
    assert audit["analytical_censoring"]==[]
    assert audit["analytical_censoring_count"]==len(records)==1
    assert records[0]["original"]=="<10"

def test_adapter_rejects_bad_factor_and_mapping():
    frame=pd.DataFrame({"a":[1],"b":[2],"c":[3]})
    with pytest.raises(ValueError,match="factor"):
        harmonize_georoc_frame(frame,GeorocAdapterConfig({"a":"Li_icpms"},lithium_scale_to_ppm=-1))
    with pytest.raises(ValueError,match="Ambiguous"):
        harmonize_georoc_frame(frame,GeorocAdapterConfig({"a":"Li_icpms","b":"Li_icpms"}))
