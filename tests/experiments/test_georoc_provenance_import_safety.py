from importlib import import_module
import pytest

@pytest.mark.parametrize("name",["data_lab.source_studies.georoc.provenance.bibliography","data_lab.source_studies.georoc.provenance.primary_li_crosswalk","data_lab.source_studies.georoc.screening.provenance_screen","data_lab.source_studies.georoc.provenance.pdf_screen","data_lab.source_studies.georoc.provenance.source_lookup","model_2.external_validation.imagery.batch"])
def test_historical_georoc_import_has_no_acquisition_or_data_read(name,tmp_path,monkeypatch):
    import pandas as pd
    import requests
    monkeypatch.chdir(tmp_path)
    def forbidden(*args,**kwargs):
        pytest.fail("Unexpected import-time I/O")
    monkeypatch.setattr(pd,"read_csv",forbidden)
    monkeypatch.setattr(pd,"read_excel",forbidden)
    monkeypatch.setattr(requests,"get",forbidden)
    module=import_module("lithiumscope.experiments."+name)
    assert callable(module.main)
    assert not list(tmp_path.iterdir())


def test_generated_final_report_stays_in_experiment_results():
    from lithiumscope.experiments.reporting import georoc_evidence as report
    assert report.DOC.is_relative_to(report.BASE)

def test_acquisition_packages_retired():
    from pathlib import Path
    root = Path(__file__).resolve().parents[2] / "src/lithiumscope/experiments"
    assert not list((root / "acquisition").rglob("*.py"))
    assert not list((root / "georoc").rglob("*.py"))

def test_network_studies_require_authorization(tmp_path, monkeypatch):
    import requests
    from lithiumscope.experiments.data_lab.source_studies.georoc.studies import single_sample_audit
    from lithiumscope.experiments.data_lab.source_studies.georoc.provenance import pdf_screen, source_lookup
    from lithiumscope.experiments.data_lab.source_studies import unab as unab_source_review
    monkeypatch.setattr(requests, "get", lambda *a, **k: pytest.fail("Unexpected network"))
    assert single_sample_audit.main()["state"] == "preflight"
    assert pdf_screen.main()["state"] == "preflight"
    assert source_lookup.main()["state"] == "preflight"
    assert unab_source_review.audit(tmp_path)["state"] == "preflight"
    assert not list(tmp_path.iterdir())
