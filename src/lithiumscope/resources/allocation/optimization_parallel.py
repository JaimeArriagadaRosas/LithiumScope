"""Trusted, temporary CPU-only scheduling of independent optimization calculations."""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import os

from lithiumscope.core.hashing import file_sha256
from lithiumscope.resources.allocation.policy import Workload, plan_workload
from lithiumscope.resources.allocation.scheduler import Task, run_tasks
from lithiumscope.resources.detection.host import detect_host


@dataclass(frozen=True)
class OptimizationExecution:
    mode: str = "serial"
    width: int = 1
    memory_bytes: int = 0


def choose_execution(device, pruning_enabled):
    from lithiumscope.resources.manager import active_fit_demand, load_policy
    demand=active_fit_demand()
    if (device is None or demand is None or os.environ.get("LITHIUMSCOPE_SCHEDULER_WORKER")
            or device.is_gpu or device.any_cuda_backend or device.mps_available or pruning_enabled):
        return OptimizationExecution()
    policy=load_policy()
    if policy.parallel_level not in {"inner_fold","trial_batch"} or policy.max_parallel_tasks<2:
        return OptimizationExecution()
    if policy.parallel_level=="trial_batch" and not policy.allow_batched_optuna:
        return OptimizationExecution()
    plan=plan_workload(detect_host(include_gpu=False),policy,Workload(
        tasks=policy.max_parallel_tasks,memory_per_task_bytes=demand.working_memory_estimate_bytes))
    if plan.parallel_tasks<2:
        return OptimizationExecution()
    return OptimizationExecution(policy.parallel_level,plan.parallel_tasks,demand.working_memory_estimate_bytes)


def evaluate_serialized_call(path, expected_sha, trial_number=None, fold_number=None):
    """Deserialize only a hash-bound closure created by this local optimizer."""
    from joblib.externals import cloudpickle
    from lithiumscope.core.logging.context import scientific_log_context
    path=Path(path)
    if not path.is_file() or file_sha256(path)!=expected_sha:
        raise ValueError("Optimization worker payload changed")
    with path.open("rb") as handle:
        fn,args=cloudpickle.load(handle)
    with scientific_log_context(trial=trial_number,fold=fold_number,stage="parallel_optuna"):
        return fn(*args)


def map_calls(function, calls, execution):
    """Ordered results, no silent retries, no nested outer pools."""
    calls=list(calls)
    if execution.mode=="serial" or len(calls)<=1:
        return [function(*args) for args,_,_ in calls]
    if os.environ.get("LITHIUMSCOPE_SCHEDULER_WORKER"):
        raise RuntimeError("Nested optimization scheduler is forbidden")
    from joblib.externals import cloudpickle
    from lithiumscope.core.logging.context import unique_id
    from lithiumscope.core.logging.session import current_session_directory
    from lithiumscope.resources.manager import load_policy
    root=current_session_directory()/"optimization"/unique_id(execution.mode)
    root.mkdir(parents=True,exist_ok=False)
    tasks,payloads=[],[]
    try:
        for i,(args,trial,fold) in enumerate(calls):
            payload=root/f"task_{i}.pickle"
            payloads.append(payload)
            with payload.open("xb") as handle:
                cloudpickle.dump((function,args),handle)
            tasks.append(Task(f"task_{i}","lithiumscope.resources.allocation.optimization_parallel:evaluate_serialized_call",
                {"path":str(payload),"expected_sha":file_sha256(payload),"trial_number":trial,"fold_number":fold},
                execution.memory_bytes))
        policy=replace(load_policy(),max_parallel_tasks=execution.width)
        result=run_tasks(tasks,root/"execution",policy=policy,execute=True)
        records={r["task_id"]:r for wave in result["waves"] for r in wave["results"]}
        if result["state"]!="completed" or len(records)!=len(tasks):
            raise RuntimeError("One or more optimization workers failed; inspect scheduler evidence")
        return [records[task.task_id]["result"] for task in tasks]
    finally:
        for payload in payloads:
            payload.unlink(missing_ok=True)
