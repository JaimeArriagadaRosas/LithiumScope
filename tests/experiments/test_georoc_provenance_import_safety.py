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


def test_generated_final_report_stays_in_experiment_results():
    from lithiumscope.experiments.georoc import georoc_final_evidence_report as report
    assert report.DOC.is_relative_to(report.BASE)

def test_remaining_georoc_scripts_use_scoped_cli_logs():
    from pathlib import Path
    root=Path(__file__).resolve().parents[2]/"src/lithiumscope/experiments/georoc"
    for name in ("analyze_georoc_background","diagnose_georoc_grouped","georoc_20576_final_gate",
                 "georoc_eligibility_gate","georoc_extended_screen","georoc_final_evidence_report",
                 "georoc_sample_method_audit","publication_first_review","unab_source_review"):
        source=(root/(name+".py")).read_text(encoding="utf-8-sig")
        assert f'run_log("simulations","{name}")' in source
    assert '--acquire' in (root/"unab_source_review.py").read_text(encoding="utf-8-sig")
