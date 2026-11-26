"""New candidates have their own annex even when historical artifacts are missing."""
import json
from types import SimpleNamespace

from lithiumscope.prediction.analysis.model_selection_evidence import gather_selection_evidence
from lithiumscope.reporting.sections.model_selection import build_model_selection
from lithiumscope.reporting.styles import build_styles


def test_current_annex_does_not_substitute_historical_metrics(tmp_path):
    one={"run_id":"training_new_1","algorithm":"random_forest","dataset_sha256":"one",
         "model_sha256":"x","training_metrics":{"rmse":42.}}
    two={"run_id":"training_new_2","algorithm":"random_forest","dataset_sha256":"two",
         "model_sha256":"y","training_metrics":{"roc_auc":.66}}
    evidence=gather_selection_evidence(one,two,root=tmp_path)
    assert evidence["state"]=="CURRENT_SELECTION_EVIDENCE"
    assert evidence["models"]["model_1"]["metrics"]=={"rmse":42.}
    assert evidence["models"]["model_2"]["metrics"]=={"roc_auc":.66}
    assert evidence["paired"]["model_1"]["state"]=="unavailable"
    json.dumps(evidence,allow_nan=False)
    ctx=SimpleNamespace(mode="demonstration",runtime={"model_selection_evidence":evidence})
    assert len(build_model_selection(ctx,build_styles()))>3


def test_recovered_m2_diagnostics_require_identity_and_snapshot(tmp_path):
    from lithiumscope.prediction.analysis.current_selection_evidence import _m2_diagnostics
    from lithiumscope.reporting.sections.current_selection import build_current_selection

    selected={"run_id":"training_new_2","model_sha256":"model","dataset_sha256":"dataset"}
    folder=tmp_path/"results/scientific_simulations/version_comparisons/run/model_2/recovered_reports"
    folder.mkdir(parents=True)
    summary={"state":"complete","samples":706,"historical":706,
             "spectral_QA_pass":690,"SCL_accepted_as_optical_proxy":566,
             "no_field_rock_certifications":True,"not_independent_external_validation":True}
    status={**selected,"training":False,"reports":{
        "scientific_diagnostics":{"state":"complete","result":summary}}}
    (folder/"recovery_status.json").write_text(json.dumps(status),encoding="utf-8")
    path=folder/"scientific_diagnostics_summary.json"
    path.write_text(json.dumps(summary),encoding="utf-8")
    evidence=_m2_diagnostics(selected,tmp_path)
    assert evidence["state"]=="complete"
    assert evidence["comparison_run"]=="run"
    assert evidence["data"]==summary
    assert _m2_diagnostics({**selected,"model_sha256":"other"},tmp_path)["state"]=="unavailable"
    path.write_text(json.dumps({**summary,"samples":783}),encoding="utf-8")
    assert _m2_diagnostics(selected,tmp_path)["state"]=="unavailable"
    path.unlink()
    # Rendering consumes the frozen snapshot, not live result files.
    elements=build_current_selection({"models":{},"paired":{},"m2_diagnostics":evidence},build_styles())
    assert any("Diagnóstico científico histórico M2" in getattr(e,"text","") for e in elements)
    json.dumps(evidence,allow_nan=False)
