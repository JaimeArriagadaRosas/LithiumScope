from importlib import import_module
import pytest

@pytest.mark.parametrize("name",["provenance.georoc_bibliography_audit","provenance.georoc_primary_li_crosswalk","provenance.georoc_provenance_screen","provenance.publication_pdf_screen","provenance.publication_source_lookup","background_georoc_exploratory"])
def test_historical_georoc_import_has_no_acquisition_or_data_read(name,tmp_path,monkeypatch):
    import pandas as pd
    import requests
    monkeypatch.chdir(tmp_path)
    def forbidden(*args,**kwargs):
        pytest.fail("Unexpected import-time I/O")
    monkeypatch.setattr(pd,"read_csv",forbidden)
    monkeypatch.setattr(pd,"read_excel",forbidden)
    monkeypatch.setattr(requests,"get",forbidden)
    module=import_module("lithiumscope.experiments.georoc."+name)
    assert callable(module.main)
    assert not list(tmp_path.iterdir())
