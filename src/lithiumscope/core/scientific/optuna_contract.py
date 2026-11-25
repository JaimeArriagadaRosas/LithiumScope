"""Shared bounded optimization contract, independent of any estimator family."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import math

import numpy as np

from lithiumscope.core.checkpoints import _atomic_json_write
from lithiumscope.core.scientific.checkpoint_identity import evaluation_identity

POLICY = "bounded_attempts_v1"
TERMINAL = {"COMPLETE", "PRUNED", "FAIL"}


def study_contract(storage_path, *, x, y, groups, splits, seed, budget, pruning,
                   direction, parameter_space, estimator_factory, model_group, extra=None, execution=None):
    if budget < 1 or int(budget) != budget:
        raise ValueError("Optuna attempt budget must be a positive integer")
    # Closure configuration is bound by the outer checkpoint identity in normal runs.
    # Standalone studies also record the explicit model parameters and caller policy.
    import inspect
    import hashlib
    import marshal

    def code_digest(fn):
        try:
            source = inspect.getsource(fn)
        except (OSError, TypeError):
            code = getattr(fn, "__code__", None)
            if code is None:
                raise ValueError("Optuna callable requires inspectable code")
            return hashlib.sha256(marshal.dumps(code)).hexdigest()
        return hashlib.sha256(source.encode()).hexdigest()

    policy = {"policy": POLICY, "seed": seed, "planned_attempts": int(budget),
              "pruning": pruning, "direction": direction,
              "space_source_sha256": code_digest(parameter_space),
              "factory_source_sha256": code_digest(estimator_factory)}
    if execution is not None and execution.mode!="serial":
        policy["optimization_execution"]={"mode":execution.mode,"batch_width":execution.width,"pruning":False}
    # No assumption that a standalone caller has a surrounding run.json.
    root = Path(storage_path).parent if storage_path else Path.cwd() / ".optuna_ephemeral"
    identity = evaluation_identity(root, x=x, y=y, groups=groups, splits=splits,
                                   policy=policy, extra=extra, model_group=model_group)
    return {**policy, "identity": identity, "expected_inner_folds": len(splits)}


def bind_study(study, contract):
    if study.direction.name.lower() != contract["direction"]:
        raise ValueError("Optuna direction mismatch")
    previous = study.user_attrs.get("evaluation_contract")
    if previous is None:
        if study.trials:
            raise ValueError("Unbound historical Optuna study cannot be resumed")
        study.set_user_attr("evaluation_contract", contract)
    elif previous != contract:
        raise ValueError("Optuna evaluation contract mismatch")


def record_fold(trial, value, step):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Nonfinite inner objective")
    trial.set_user_attr("evaluated_inner_folds", step + 1)
    trial.set_user_attr(f"fold_{step + 1}_metric", value)


def summarize_study(study, contract):
    trials = study.get_trials(deepcopy=False)
    states = Counter(t.state.name for t in trials)
    def complete_evidence(t):
        if t.state.name!="COMPLETE" or t.value is None or not math.isfinite(t.value):
            return False
        n=contract["expected_inner_folds"]
        if t.user_attrs.get("evaluated_inner_folds")!=n:
            return False
        metrics=[t.user_attrs.get(f"fold_{i}_metric") for i in range(1,n+1)]
        return all(isinstance(v,(int,float)) and math.isfinite(v) for v in metrics) and math.isclose(
            t.value,sum(metrics)/n,rel_tol=1e-12,abs_tol=1e-12)
    complete = [t for t in trials if complete_evidence(t)]
    terminal = sum(states[s] for s in TERMINAL)
    best = None
    if complete:
        best = min(complete, key=lambda t: (t.value if contract["direction"] == "minimize" else -t.value, t.number))
    return {"contract": contract, "study_name": study.study_name,
            "planned_attempts": contract["planned_attempts"], "attempted": len(trials),
            "terminal": terminal, "completed": states["COMPLETE"], "complete_valid": len(complete),
            "pruned": states["PRUNED"], "failed": states["FAIL"],
            "running": states["RUNNING"], "waiting": states["WAITING"],
            "evaluated_inner_folds": sum(int(t.user_attrs.get("evaluated_inner_folds", 0)) for t in trials),
            "planned_inner_fits_upper_bound": contract["planned_attempts"] * contract["expected_inner_folds"],
            "best_trial": best.number if best is not None else None,
            "best_value": best.value if best is not None else None,
            "best_params": dict(best.params) if best is not None else None,
            "budget_exhausted": len(trials) >= contract["planned_attempts"],
            "sufficiency_certified": False}


def run_bounded_study(study, objective, contract, *, storage_path=None, progress_callback=None, batch_runner=None):
    bind_study(study, contract)
    budget = contract["planned_attempts"]

    def snapshot():
        report = summarize_study(study, contract)
        if storage_path:
            path = Path(storage_path).with_name(Path(storage_path).stem+"."+contract["identity"][:16]+".audit.json")
            _atomic_json_write(path, report)
        return report

    try:
        report = snapshot()
        # Do not assume that a RUNNING trial belongs to a dead process. It must
        # be reconciled explicitly rather than overwritten or counted as complete.
        if report["running"] or report["waiting"]:
            raise RuntimeError("Optuna study has unresolved RUNNING/WAITING trials")
        if report["attempted"] > budget:
            raise ValueError("Optuna study exceeds declared attempt budget")
        if progress_callback:
            progress_callback(report["terminal"], budget)

        def on_terminal(study, trial):
            current = snapshot()
            if progress_callback:
                progress_callback(current["terminal"], budget)

        remaining = budget - report["attempted"]
        if remaining:
            # Estimator failures remain FAIL and propagate (no infinite retries).
            if batch_runner is None:
                study.optimize(objective, n_trials=remaining, callbacks=[on_terminal],
                               show_progress_bar=False, n_jobs=1)
            else:
                batch_runner(study,remaining,on_terminal)
        report = snapshot()
        if report["complete_valid"] != report["completed"]:
            raise RuntimeError("Optuna COMPLETE trial lacks finite full-fold evidence")
        if not report["complete_valid"]:
            raise RuntimeError("Optuna study has no completed trial.")
        return report
    finally:
        snapshot()


def validate_splits(splits, x, y, groups=None):
    from lithiumscope.core.scientific_checks import assert_partition_coverage, assert_group_isolation
    splits = [(np.asarray(tr), np.asarray(va)) for tr, va in splits]
    if len(x) != len(y) or not splits:
        raise ValueError("Invalid optimization cohort or empty CV")
    if any(not np.issubdtype(v.dtype, np.integer) for pair in splits for v in pair):
        raise ValueError("Noninteger optimization indices")
    assert_partition_coverage(splits, len(y))
    if hasattr(x,"index") and hasattr(y,"index") and not x.index.equals(y.index):
        raise ValueError("Misaligned optimization target index")
    if groups is not None:
        import pandas as pd
        if not isinstance(groups,pd.Series):
            groups=pd.Series(groups)
        if groups.isna().any() or groups.astype(str).str.strip().eq("").any():
            raise ValueError("Missing optimization group identity")
        if len(groups) != len(y):
            raise ValueError("Misaligned optimization groups")
        if hasattr(x, "index") and hasattr(groups, "index") and not x.index.equals(groups.index):
            raise ValueError("Misaligned optimization group index")
        for tr, va in splits:
            assert_group_isolation(tr, va, groups)
    return splits
