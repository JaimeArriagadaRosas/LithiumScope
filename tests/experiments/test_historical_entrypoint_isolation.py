from importlib import import_module
import pytest

@pytest.mark.parametrize("name", ["lithiumscope.experiments.model_1.run_vnext_stage1","lithiumscope.experiments.integrated.academic_evidence","lithiumscope.experiments.model_2.seasonal_comparison","lithiumscope.experiments.model_2.sensitivity_analysis","lithiumscope.experiments.model_2.temporal_multiscene_analysis","lithiumscope.experiments.model_2.audit_georoc_m2_candidate_coverage","lithiumscope.experiments.model_2.audit_georoc_source_references","lithiumscope.experiments.model_2.pilot_georoc_m2_labels"])
def test_canonical_experiment_import(name):
    module = import_module(name)
    assert module.__name__ == name

@pytest.mark.parametrize("old,new", [["lithiumscope.experiments.validation.spatial_batch_worker","lithiumscope.tools.acquisition.sentinel2.spatial_batch_worker"]])
def test_experimental_relocation(old,new):
    assert import_module(old) is import_module(new)

@pytest.mark.parametrize("name",["studies.andean_partitions","screening.andean_candidates","imagery.pilot"])
def test_historical_import_does_not_acquire_or_read_data(name,monkeypatch,tmp_path):
    monkeypatch.chdir(tmp_path)
    import requests
    import pandas as pd
    def forbidden(*args,**kwargs):
        pytest.fail("Import attempted acquisition or dataset read")
    monkeypatch.setattr(requests,"get",forbidden)
    monkeypatch.setattr(pd,"read_csv",forbidden)
    module=import_module("lithiumscope.experiments.georoc."+name)
    assert callable(module.main)
    assert not list(tmp_path.iterdir())
