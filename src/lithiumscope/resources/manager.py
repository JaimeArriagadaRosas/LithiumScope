"""Central resource interface used by training, diagnostics and reproducibility."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict
from pathlib import Path

from lithiumscope.core.config import load_config
from lithiumscope.resources.allocation.policy import ResourcePolicy, Workload, plan_workload
from lithiumscope.resources.detection.host import detect_host
from lithiumscope.resources.monitoring.monitor import monitor_operation

_active_threads: ContextVar[int | None] = ContextVar("resource_threads", default=None)


def load_policy() -> ResourcePolicy:
    return ResourcePolicy.from_config(load_config("resources"))


def cpu_worker_budget() -> int:
    active = _active_threads.get()
    if active is not None:
        return active
    return plan_workload(detect_host(), load_policy(), Workload()).threads_per_task


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
    from threadpoolctl import threadpool_limits

    policy = load_policy()
    plan = plan_workload(detect_host(), policy, workload or Workload())
    token = _active_threads.set(plan.threads_per_task)
    try:
        with monitor_operation(path, interval_seconds=policy.monitor_interval_seconds):
            with threadpool_limits(limits=plan.threads_per_task):
                yield plan
    finally:
        _active_threads.reset(token)
