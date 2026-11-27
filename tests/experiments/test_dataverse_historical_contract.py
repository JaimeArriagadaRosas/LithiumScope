import pandas as pd
import pytest
from lithiumscope.datasets.adapters.georoc.dataverse import whole_rock_candidates
from lithiumscope.tools.acquisition.georoc.dataverse import fetch_partition
from lithiumscope.experiments.data_lab.source_studies.georoc.studies.andean_partitions import run

def test_candidate_adapter_excludes_nonfinite_and_nonrock():
    frame=pd.DataFrame({"LATITUDE MIN":[-20]*3,"LONGITUDE MIN":[-70]*3,
        "LI(PPM)":[4,float("inf"),6],"MATERIAL":["WR","WR","BRINE"],"SAMPLE NAME":["a","b","c"],
        "CITATIONS":["ref"]*3,"LOCATION":["Chile"]*3})
    candidates,audit=whole_rock_candidates(frame,"synthetic.csv")
    assert candidates.sample_name.tolist()==["a"] and audit["candidate_whole_rock"]==1

def test_historical_download_preflight_does_not_create_files(tmp_path):
    out=tmp_path/"raw"
    assert run(out)["state"]=="preflight"
    assert not out.exists()

def test_invalid_cached_partition_is_not_overwritten(tmp_path):
    path=tmp_path/"source.csv"
    path.write_bytes(b"partial")
    with pytest.raises(ValueError,match="incomplete"):
        fetch_partition(1,path,request=lambda *a,**k:pytest.fail("unexpected network"))
    assert path.read_bytes()==b"partial"
