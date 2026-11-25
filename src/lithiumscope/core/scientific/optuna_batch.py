"""Explicit batched search: parent owns Optuna storage and deterministic tell order."""
from __future__ import annotations

import math


def run_batched_trials(study, remaining, on_terminal, *, parameter_space, score_trial, execution, expected_folds):
    """Batch TPE differs from sequential TPE; no pruning, no hidden retries."""
    import optuna
    from lithiumscope.core.scientific.optuna_contract import record_fold
    from lithiumscope.resources.allocation.optimization_parallel import map_calls
    if execution.mode!="trial_batch" or execution.width<2:
        raise ValueError("Explicit batched search policy required")
    while remaining:
        pending={}
        try:
            # Suggest in parent; workers never access SQLite or the Optuna trial object.
            for _ in range(min(execution.width,remaining)):
                trial=study.ask()
                pending[trial.number]=(trial,None)
                pending[trial.number]=(trial,parameter_space(trial))
            calls=[((params,),number,None) for number,(_,params) in pending.items()]
            values=map_calls(score_trial,calls,execution)
            for number,metrics in zip(list(pending),values,strict=True):
                trial,_=pending[number]
                if len(metrics)!=expected_folds or not all(math.isfinite(float(v)) for v in metrics):
                    raise ValueError("Incomplete or nonfinite batched trial")
                for step,metric in enumerate(metrics):
                    record_fold(trial,metric,step)
                    trial.report(sum(metrics[:step+1])/(step+1),step=step)
                frozen=study.tell(trial,sum(metrics)/len(metrics))
                del pending[number]
                remaining-=1
                on_terminal(study,frozen)
        except BaseException:
            # Explicitly record all already-asked but untold attempts as failed.
            for trial,_ in pending.values():
                frozen=study.tell(trial,state=optuna.trial.TrialState.FAIL)
                on_terminal(study,frozen)
            raise
