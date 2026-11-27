from importlib import import_module
from importlib.util import find_spec
import pytest

@pytest.mark.parametrize("name", ["lithiumscope.experiments.model_1.studies.data_inspection","lithiumscope.experiments.integrated.academic_evidence","lithiumscope.experiments.model_2.temporal.seasonal","lithiumscope.experiments.model_2.evaluation.sensitivity","lithiumscope.experiments.model_2.temporal.multiscene","lithiumscope.experiments.model_2.external_validation.coordinate_coverage","lithiumscope.experiments.model_2.external_validation.source_references","lithiumscope.experiments.model_2.external_validation.label_candidates"])
def test_canonical_experiment_import(name):
    module = import_module(name)
    assert module.__name__ == name

@pytest.mark.parametrize("old,new", [["lithiumscope.experiments.validation.spatial_batch_worker","lithiumscope.tools.acquisition.sentinel2.spatial_batch_worker"]])
def test_experimental_relocation(old,new):
    assert find_spec(old) is None
    assert import_module(new).__name__ == new

@pytest.mark.parametrize("name",["data_lab.source_studies.georoc.studies.andean_partitions","data_lab.source_studies.georoc.screening.andean_candidates","model_2.external_validation.imagery.pilot"])
def test_historical_import_does_not_acquire_or_read_data(name,monkeypatch,tmp_path):
    monkeypatch.chdir(tmp_path)
    import requests
    import pandas as pd
    def forbidden(*args,**kwargs):
        pytest.fail("Import attempted acquisition or dataset read")
    monkeypatch.setattr(requests,"get",forbidden)
    monkeypatch.setattr(pd,"read_csv",forbidden)
    module=import_module("lithiumscope.experiments."+name)
    assert callable(module.main)
    assert not list(tmp_path.iterdir())
