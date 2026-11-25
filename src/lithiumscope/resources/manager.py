"""Central resource interface used by training, diagnostics and reproducibility."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict
from pathlib import Path
from threading import RLock

from lithiumscope.resources.allocation.runtime import bounded_workers, bounded_cache_mb, native_thread_limits

from lithiumscope.core.config import load_config
from lithiumscope.resources.allocation.policy import ResourcePolicy, Workload, plan_workload
from lithiumscope.resources.detection.host import detect_host
from lithiumscope.resources.monitoring.monitor import monitor_operation

_active_threads: ContextVar[int | None] = ContextVar("resource_threads", default=None)
_active_plan = ContextVar("resource_plan",default=None)
_native_lock=RLock()


def load_policy() -> ResourcePolicy:
    return ResourcePolicy.from_config(load_config("resources"))


def cpu_worker_budget() -> int:
    active = _active_threads.get()
    if active is not None:
        return active
    return plan_workload(detect_host(include_gpu=False), load_policy(), Workload()).threads_per_task


def worker_limit(requested=None):
    return bounded_workers(requested,cpu_worker_budget())


def svm_cache_limit(requested=200.):
    plan=_active_plan.get()
    if plan is None:
        plan=plan_workload(detect_host(include_gpu=False),load_policy(),Workload())
    return bounded_cache_mb(requested,plan.ram_budget_bytes)


@contextmanager
def estimator_thread_scope():
    # Process-global native thread settings must not overlap across threads.
    with _native_lock:
        with native_thread_limits(cpu_worker_budget()):
            yield


@contextmanager
def fit_workload_scope(demand, *, evidence_path=None):
    """Reassess available RAM/CPU for this algorithm without nested parallelism."""
    policy=load_policy()
    plan=plan_workload(detect_host(include_gpu=False),policy,demand.workload())
    parent=_active_threads.get()
    threads=min(plan.threads_per_task,parent) if parent is not None else plan.threads_per_task
    from dataclasses import replace
    plan=replace(plan,threads_per_task=threads)
    token=_active_threads.set(threads)
    plan_token=_active_plan.set(plan)
    try:
        if evidence_path is not None:
            import json
            from uuid import uuid4
            evidence_path.parent.mkdir(parents=True,exist_ok=True)
            temporary=evidence_path.with_name(f".{evidence_path.name}.{uuid4().hex}.tmp")
            try:
                temporary.write_text(json.dumps({"demand":demand.to_dict(),"plan":asdict(plan),
                    "estimate_is_hard_cap":False},indent=2),encoding="utf-8")
                temporary.replace(evidence_path)
            finally:
                temporary.unlink(missing_ok=True)
        with estimator_thread_scope():
            yield plan
    finally:
        _active_plan.reset(plan_token)
        _active_threads.reset(token)


def resource_snapshot(policy: ResourcePolicy | None = None) -> dict:
    host = detect_host()
    effective = policy or load_policy()
    plan = plan_workload(host, effective, Workload())
    return {"cpu_count": host.logical_cpus, "cpu_worker_budget": plan.cpu_budget,
            **asdict(host), "policy": asdict(effective), "plan": asdict(plan),
            "nvidia_detected": bool(host.gpus),
            "cuda_available": None,  # hardware inventory does not certify a library backend
            }


@contextmanager
def resource_operation(path: Path, workload: Workload | None = None):

    policy = load_policy()
    plan = plan_workload(detect_host(), policy, workload or Workload())
    token = _active_threads.set(plan.threads_per_task)
    plan_token=_active_plan.set(plan)
    try:
        with monitor_operation(path, interval_seconds=policy.monitor_interval_seconds,
                               metadata={"plan":asdict(plan),"policy":asdict(policy),
                                         "limits":"CPU supported native libraries; RAM/VRAM estimates, not OS hard caps"}):
            with estimator_thread_scope():
                yield plan
    finally:
        _active_plan.reset(plan_token)
        _active_threads.reset(token)
