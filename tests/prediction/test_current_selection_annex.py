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
