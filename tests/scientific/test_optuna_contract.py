import json
import numpy as np
import pandas as pd
import pytest
import optuna
from sklearn.model_selection import GroupKFold
from sklearn.dummy import DummyRegressor

from lithiumscope.core.scientific.optuna_contract import record_fold, run_bounded_study
from lithiumscope.model_1.training.optimization.optimizer import optimize_with_optuna


def contract(budget=3):
    return {"planned_attempts":budget,"expected_inner_folds":2,
            "direction":"minimize","identity":"a"*64}


def test_bounded_attempts_and_pruning(tmp_path):
    study=optuna.create_study(direction="minimize")
    def objective(trial):
        record_fold(trial,1.,0)
        if trial.number==1:
            raise optuna.TrialPruned()
        record_fold(trial,2.,1)
        return 1.5
    result=run_bounded_study(study,objective,contract(),storage_path=tmp_path/"s.db")
    assert (result["attempted"],result["complete_valid"],result["pruned"])==(3,2,1)
    assert result["evaluated_inner_folds"]==5
    assert result["best_trial"]==0 and not result["sufficiency_certified"]
    assert len(list(tmp_path.glob("*.audit.json")))==1
    assert run_bounded_study(study,objective,contract())["attempted"]==3
    with pytest.raises(ValueError,match="contract mismatch"):
        run_bounded_study(study,objective,{**contract(),"identity":"different"})


def test_failed_trial_is_not_retried_or_counted_as_complete(tmp_path):
    study=optuna.create_study(direction="minimize")
    def objective(trial):
        raise RuntimeError("fit failed")
    with pytest.raises(RuntimeError,match="fit failed"):
        run_bounded_study(study,objective,contract(),storage_path=tmp_path/"s.db")
    report=json.loads(next(tmp_path.glob("*.audit.json")).read_text())
    assert report["failed"]==1 and report["complete_valid"]==0 and report["attempted"]==1


def test_running_and_legacy_studies_fail_closed():
    study=optuna.create_study(direction="minimize")
    study.set_user_attr("evaluation_contract",contract())
    study.ask()
    with pytest.raises(RuntimeError,match="RUNNING"):
        run_bounded_study(study,lambda t:0.,contract())
    legacy=optuna.create_study(direction="minimize")
    legacy.optimize(lambda t:1.,n_trials=1)
    with pytest.raises(ValueError,match="Unbound historical"):
        run_bounded_study(legacy,lambda t:0.,contract())


def test_full_fold_evidence_required():
    study=optuna.create_study(direction="minimize")
    with pytest.raises(RuntimeError,match="lacks finite full-fold evidence"):
        run_bounded_study(study,lambda t:1.,contract(1))


def factory(params):
    return DummyRegressor(**params)


def space(trial):
    return {"strategy":trial.suggest_categorical("strategy",["mean","median"])}


def test_real_sqlite_resume_refuses_changed_data(tmp_path):
    x=pd.DataFrame({"x":np.arange(12.)})
    y=pd.Series(np.arange(12.))
    g=pd.Series(np.repeat(["a","b","c","d"],3))
    splits=list(GroupKFold(2).split(x,y,g))
    kwargs=dict(n_trials=3,storage_path=tmp_path/"study.db",study_name="audit",groups=g,pruning_enabled=False)
    first=optimize_with_optuna(factory,space,x,y,splits,**kwargs)
    second=optimize_with_optuna(factory,space,x,y,splits,**kwargs)
    assert first==second
    report=json.loads(next(tmp_path.glob("*.audit.json")).read_text())
    assert report["attempted"]==report["complete_valid"]==3
    with pytest.raises(ValueError,match="contract mismatch"):
        optimize_with_optuna(factory,space,x,y+1,splits,**kwargs)


def test_invalid_fold_partition_is_rejected_before_fit():
    x=pd.DataFrame({"x":np.arange(4.)})
    y=pd.Series(np.arange(4.))
    with pytest.raises(ValueError):
        optimize_with_optuna(factory,space,x,y,[(np.array([0,1]),np.array([1,2]))],1)
