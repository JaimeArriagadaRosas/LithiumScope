from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from lithiumscope.model_1.evaluation.baseline import evaluate_baselines, evaluate_baseline
from lithiumscope.model_1.evaluation.metrics import regression_metrics
from lithiumscope.model_1.evaluation.selection import baseline_eligibility


def prepared():
    idx=pd.Index([10,11,20,21,30,31])
    return SimpleNamespace(x=pd.DataFrame({"x":[1,2,3,4,5,6]},index=idx),
        y=pd.Series([1.,2.,3.,4.,100.,200.],index=idx),
        groups=pd.Series(["a","a","b","b","c","c"],index=idx),
        frame=pd.DataFrame({"source_sample":list("abcdef")},index=idx),
        outer_splits=[(np.array([2,3,4,5]),np.array([0,1])),
                      (np.array([0,1,4,5]),np.array([2,3])),
                      (np.array([0,1,2,3]),np.array([4,5]))],
        validation_strategy="group_kfold_spatial")


CONFIG={"validation":{"outer_folds":3,"random_seed":42,"prefer_spatial_groups":True}}


def test_both_baselines_identified_and_train_only():
    p=prepared()
    rows,folds,oof=evaluate_baselines(p,CONFIG)
    assert len(rows)==2 and len(folds)==6 and len(oof)==12
    first=oof[(oof.algorithm=="baseline_mean")&(oof.outer_fold==1)]
    assert first.Li_icpms_predicted.tolist()==[76.75,76.75]
    assert first.row_index.tolist()==[10,11]
    assert first.spatial_group.tolist()==["a","a"]
    assert first.fold_train_q75_ppm.tolist()==[125.,125.]
    med=oof[(oof.algorithm=="baseline_median")&(oof.outer_fold==1)]
    assert med.Li_icpms_predicted.tolist()==[52.,52.]
    p.y.iloc[:2]=10000.
    _,_,changed=evaluate_baseline(p,CONFIG,strategy="mean")
    np.testing.assert_array_equal(changed[:2],first.Li_icpms_predicted)


def test_baseline_requires_partition_and_isolated_groups():
    p=prepared()
    p.outer_splits[0]=(np.array([0,3,4,5]),np.array([1,2]))
    with pytest.raises(ValueError):
        evaluate_baselines(p,CONFIG)


def test_constant_truth_is_explicitly_undefined_not_perfect():
    m=regression_metrics([3.,3.],[3.,3.])
    assert m.rmse==m.mae==0
    assert np.isnan(m.r2) and np.isnan(m.explained_variance)
    assert np.isnan(regression_metrics([1.],[1.]).r2)


@pytest.mark.parametrize("y,p", [([1.,2.],[[1.],[2.]]),([1.,np.nan],[1.,2.]),([],[]),([1.],[1.,2.])])
def test_invalid_regression_arrays_fail(y,p):
    with pytest.raises(ValueError,match="aligned"):
        regression_metrics(y,p)


def test_relative_rank_does_not_imply_scientific_eligibility():
    rows=pd.DataFrame([
        {"algorithm":"relative_best","is_baseline":False,"status":"ok","rmse_mean":5.},
        {"algorithm":"equal","is_baseline":False,"status":"ok","rmse_mean":6.},
        {"algorithm":"ineligible","is_baseline":False,"status":"ok","rmse_mean":1.,"eligible_for_selection":False},
        {"algorithm":"baseline_mean","is_baseline":True,"status":"ok","rmse_mean":7.},
        {"algorithm":"baseline_median","is_baseline":True,"status":"ok","rmse_mean":6.},
    ])
    out=baseline_eligibility(rows,preflight_ok=True)
    assert out.scientific_candidate_eligible.tolist()==[True,False,False,False,False]
    assert not out.release_eligible.any()
    assert not baseline_eligibility(rows,preflight_ok=False).scientific_candidate_eligible.any()
    assert not baseline_eligibility(rows,preflight_ok=True,minimum_gain=1.).scientific_candidate_eligible.any()
    with pytest.raises(ValueError,match="mandatory"):
        baseline_eligibility(rows.iloc[:-1],preflight_ok=True)



def test_finalizer_does_not_fit_when_no_candidate_beats_baseline(tmp_path,monkeypatch):
    from lithiumscope.model_1.training import competition
    from lithiumscope.model_1.training.execution.competition_finalize_model_1_winner import _finalize_model_1_winner
    from lithiumscope.core.states import RunState

    p=prepared()
    baseline=evaluate_baselines(p,CONFIG)
    baseline[0][0]["rmse_mean"]=1.
    baseline[0][1]["rmse_mean"]=2.
    rows=[*baseline[0],{"algorithm":"svm__identity","label":"SVM","is_baseline":False,
                       "status":"ok","eligible_for_selection":True,"rmse_mean":3.,"mae_mean":2.,"r2_mean":.1}]
    events=[]
    context=SimpleNamespace(root=tmp_path,tables=tmp_path,exports=tmp_path,
                            tracker=SimpleNamespace(set_state=lambda state,**kw:events.append((state,kw))))
    monkeypatch.setattr(competition,"_excel_sheets",lambda *a:{})
    monkeypatch.setattr(competition,"export_workbook",lambda *a:None)
    monkeypatch.setattr(competition,"build_dashboard",lambda *a:None)
    def forbidden(*a,**kw):
        raise AssertionError("No final fit allowed")
    monkeypatch.setattr(competition,"build_pipeline",forbidden)
    outcome=_finalize_model_1_winner(baseline,CONFIG,context,None,None,[],None,
        {"status":"pass"},p,{},rows,42)
    assert outcome.winner is None
    assert outcome.successful==["svm__identity"]
    assert events[0][0]==RunState.PARTIAL
    assert events[0][1]["relative_winner"]=="svm__identity"
    assert (tmp_path/"competition_ranking.csv").is_file()


def test_cv_runner_records_train_metrics_and_uses_bound_resume(tmp_path,monkeypatch):
    from sklearn.dummy import DummyRegressor
    from lithiumscope.model_1.training.execution import cv_runner
    from lithiumscope.resources.detection.device import DeviceInfo
    from lithiumscope.model_1.training.preprocessing.tabular import FeatureSchema
    p=prepared()
    p.schema=FeatureSchema(["x"],[],2)
    config={**CONFIG,"optimization":{"enabled":False}}
    calls=[]
    def factory(*a,**kw):
        calls.append(1)
        return DummyRegressor(strategy="mean")
    monkeypatch.setattr(cv_runner,"build_pipeline",factory)
    device=DeviceInfo("cpu","CPU",1,False,False,False)
    first=cv_runner.run_nested_cv(p,"svm",device,config,checkpoint_root=tmp_path)
    assert len(calls)==3 and "train_rmse" in first.fold_table
    assert first.fold_table.train_rmse.notna().all()
    second=cv_runner.run_nested_cv(p,"svm",device,config,checkpoint_root=tmp_path)
    assert len(calls)==3 and second.fold_table.resumed.all()
    np.testing.assert_array_equal(first.predictions,second.predictions)
    p.y.iloc[0]+=1.
    with pytest.raises(ValueError,match="different evaluation identity"):
        cv_runner.run_nested_cv(p,"svm",device,config,checkpoint_root=tmp_path)
