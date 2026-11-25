from functools import partial
import optuna
import pytest
from lithiumscope.core.scientific.optuna_batch import run_batched_trials
from lithiumscope.core.scientific.optuna_contract import run_bounded_study
from lithiumscope.resources.allocation import optimization_parallel
from lithiumscope.resources.allocation.optimization_parallel import OptimizationExecution

def test_batched_parent_preserves_attempt_budget_and_full_fold_evidence(monkeypatch):
    monkeypatch.setattr(optimization_parallel,"map_calls",lambda fn,calls,execution:[fn(*args) for args,_,_ in calls])
    study=optuna.create_study(direction="minimize")
    contract={"planned_attempts":3,"expected_inner_folds":2,"direction":"minimize","identity":"batch"}
    runner=partial(run_batched_trials,parameter_space=lambda t:{"a":t.suggest_int("a",1,2)},
        score_trial=lambda p:[float(p["a"]),float(p["a"]+1)],execution=OptimizationExecution("trial_batch",2,1),
        expected_folds=2)
    report=run_bounded_study(study,lambda _:pytest.fail("serial objective used"),contract,batch_runner=runner)
    assert report["attempted"]==report["complete_valid"]==3 and report["running"]==0
    assert report["evaluated_inner_folds"]==6

def test_batched_worker_failure_terminalizes_pending_trials(monkeypatch):
    def fail(*args): raise RuntimeError("worker failed")
    monkeypatch.setattr(optimization_parallel,"map_calls",fail)
    study=optuna.create_study(direction="minimize")
    with pytest.raises(RuntimeError,match="worker failed"):
        run_batched_trials(study,3,lambda *_:None,parameter_space=lambda t:{},
                           score_trial=lambda p:[1.,2.],execution=OptimizationExecution("trial_batch",2,1),
                           expected_folds=2)
    assert [t.state.name for t in study.trials]==["FAIL","FAIL"]
