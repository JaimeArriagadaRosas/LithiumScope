import json
import time

import pytest

from lithiumscope.resources.allocation.policy import ResourcePolicy, Workload, plan_workload
from lithiumscope.resources.detection.host import HostResources
from lithiumscope.resources.monitoring.monitor import monitor_operation
from lithiumscope.resources.manager import cpu_worker_budget


def host(ram=8_000, threads=8):
    return HostResources(logical_cpus=16, physical_cores=8, available_threads=threads,
                         ram_total_bytes=16_000, ram_available_bytes=ram, disk_free_bytes=100_000,
                         gpus=())


def test_resource_plan_limits_nested_parallelism_and_ram():
    policy = ResourcePolicy(max_cpu_fraction=.75, max_parallel_tasks=4, max_ram_fraction=.5)
    plan = plan_workload(host(), policy, Workload(tasks=10, memory_per_task_bytes=2_000))
    assert plan.cpu_budget == 6
    assert plan.parallel_tasks == 2
    assert plan.threads_per_task == 3
    assert plan.parallel_tasks * plan.threads_per_task <= plan.cpu_budget
    assert plan.parallel_tasks * 2_000 <= plan.ram_budget_bytes


def test_cpu_budget_respects_available_affinity_even_with_large_minimum():
    policy = ResourcePolicy(max_cpu_fraction=.75, minimum_workers=99)
    assert plan_workload(host(threads=2), policy, Workload()).cpu_budget == 2


def test_impossible_memory_and_gpu_requirements_block():
    policy = ResourcePolicy(max_ram_fraction=.5)
    with pytest.raises(ValueError, match="RAM"):
        plan_workload(host(), policy, Workload(memory_per_task_bytes=5_000))
    with pytest.raises(ValueError, match="GPU"):
        plan_workload(host(), policy, Workload(require_gpu=True))


def test_monitor_produces_timed_evidence(tmp_path):
    path = tmp_path / "resource.jsonl"
    with monitor_operation(path, interval_seconds=.01):
        time.sleep(.03)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows[0]["event"] == "start"
    assert rows[-1]["event"] == "finish"
    assert rows[-1]["duration_seconds"] > 0
    assert any(row["event"] == "sample" for row in rows)
    assert cpu_worker_budget() >= 1
