"""Deterministic budget checks; no machine-dependent timing threshold."""
from lithiumscope.resources.allocation.policy import ResourcePolicy, Workload, plan_workload
from lithiumscope.resources.detection.host import HostResources


def test_worker_budget_invariants():
    host=HostResources(16,8,8,16000,8000,100000,())
    for workers in (1,2,4,8):
        plan=plan_workload(host,ResourcePolicy(max_parallel_tasks=workers),Workload(tasks=8,memory_per_task_bytes=1000))
        assert plan.parallel_tasks*plan.threads_per_task<=plan.cpu_budget
        assert plan.parallel_tasks*1000<=plan.ram_budget_bytes
