"""Acquisition, overlap screening and separate lifecycle contracts."""
from pathlib import Path

import pandas as pd
import pytest

from lithiumscope.core.hashing import file_sha256
from lithiumscope.datasets.management.research_sources import ResearchSource, ResearchPartition
from lithiumscope.tools.acquisition.georoc.research import acquire
from lithiumscope.experiments.data_lab.candidate_eda import overlap_flags
from lithiumscope.experiments.data_lab import workflow


def test_registered_acquisition_reuses_bytes_and_rejects_changes(tmp_path):
    spec=ResearchSource("test","10.1/test",(ResearchPartition(123,"part.csv"),))
    calls=[]
    def download(identifier,path,*,expected_sha256=None):
        calls.append((identifier,path.exists()))
        if not path.exists():
            path.write_bytes(b"sample\n"*200)
        return {"sha256":file_sha256(path),"state":"cached" if calls[-1][1] else "downloaded"}
    files,_=acquire(spec,tmp_path,downloader=download)
    acquire(spec,tmp_path,downloader=download)
    assert calls==[(123,False),(123,True)]
    files[0].write_bytes(b"corruption")
    with pytest.raises(ValueError,match="changed"):
        acquire(spec,tmp_path,downloader=download)


def test_candidate_overlap_is_a_flag_not_training_selection():
    base=pd.DataFrame({"source_sample":["A"],"source_unique_id":["1"],"Longitude":[-70.],
                      "Latitude":[-30.],"Li_icpms":[700.]})
    new=pd.concat([base,base.assign(source_sample="B",source_unique_id="2",Li_icpms=1000.)],ignore_index=True)
    flags=overlap_flags(new,base)
    assert flags.coordinate_li_overlap.tolist()==[True,False]
    assert flags.source_unique_id_overlap.tolist()==[True,False]


def test_eda_does_not_prepare_treatments(monkeypatch,tmp_path):
    from lithiumscope.model_1.eda import pipeline
    evidence=tmp_path/"eda"
    evidence.mkdir()
    (evidence/"manifest.json").write_text('{"state":"completed"}',encoding="utf-8")
    from lithiumscope.experiments.data_lab import eda_cache
    monkeypatch.setattr(eda_cache,"lookup",lambda *_:None)
    monkeypatch.setattr(eda_cache,"key_for",lambda *_:"test")
    monkeypatch.setattr(eda_cache,"record",lambda *_:None)
    monkeypatch.setattr(pipeline,"run_eda",lambda source:evidence)
    path,state,record=workflow.analyze("model_1",Path("source.csv"),include_candidates=False)
    assert path==evidence and state=="complete"
    assert not record["treatment_plans_created"] and record["model_fits"]==0


def test_cached_eda_skips_preparation(monkeypatch,tmp_path):
    from lithiumscope.experiments.data_lab import eda_cache
    from lithiumscope.model_1.eda import pipeline
    (tmp_path/"manifest.json").write_text('{"state":"completed"}',encoding="utf-8")
    monkeypatch.setattr(eda_cache,"lookup",lambda *_:tmp_path)
    monkeypatch.setattr(pipeline,"run_eda",lambda *_:pytest.fail("No repeat EDA"))
    path,state,_=workflow.analyze("model_1",Path("source.csv"),include_candidates=False)
    assert path==tmp_path and state=="complete"
