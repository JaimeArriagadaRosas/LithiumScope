"""Scientific report numbers originate in analysis, not the renderer."""
import pandas as pd
from lithiumscope.prediction.analysis.report_inputs import prepare_cases, prepare_selection_evidence


def test_report_inputs_are_precomputed_and_source_frames_unchanged():
    one=pd.DataFrame({"Li_icpms_interval_low_q90":[-3.],"Li_icpms_interval_high_q90":[8.]})
    two=pd.DataFrame({"prospectivity_score":[.8],"operating_threshold":[.7]})
    paired=pd.DataFrame({"Li_icpms":[10.],"Li_icpms_predicted":[8.],"prospectivity_score":[.8]})
    a,b,c=prepare_cases(one,two,paired,.9)
    assert a.report_interval_low.tolist()==[0.]
    assert b.report_operating_class.tolist()==["POSITIVA"]
    assert c.report_operating_class.tolist()==["NEGATIVA"]
    assert c.report_absolute_error.tolist()==[2.]
    assert "report_absolute_error" not in paired


def test_zero_reference_gain_is_undefined_and_snapshot_is_preserved():
    evidence={"state":"HISTORICAL_COMPARISON","model_1_paired":{
      "common":{"old":{"rmse":0,"mae":4},"new":{"rmse":0,"mae":2}},
      "high_lithium":{"old":{"rmse":4},"new":{"rmse":2}}},
      "model_2_same_706_comparison":[
        {"algorithm":"original_release_protocol_retrained","roc_auc":.6,"average_precision":.5,"brier":.3},
        {"algorithm":"random_forest","roc_auc":.7,"average_precision":.6,"brier":.2}]}
    out=prepare_selection_evidence(evidence)
    assert out["report_m1_gains"]=={"rmse":None,"mae":.5,"high_rmse":.5}
    assert abs(out["report_m2_deltas"]["brier"]+.1)<1e-12
    assert "report_m1_gains" not in evidence
