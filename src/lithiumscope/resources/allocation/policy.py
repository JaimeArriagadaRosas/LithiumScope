"""Deterministic budget planning; no implicit nested parallelism."""
from __future__ import annotations

from dataclasses import dataclass
import math

from lithiumscope.resources.detection.host import HostResources


@dataclass(frozen=True)
class ResourcePolicy:
    max_cpu_fraction: float = .75
    minimum_workers: int = 1
    max_parallel_tasks: int = 1
    max_ram_fraction: float = .60
    max_vram_fraction: float = .75
    monitor_interval_seconds: float = 2.0
    parallel_level: str = "algorithm"
    allow_batched_optuna: bool = False

    def __post_init__(self):
        if self.parallel_level not in {"estimator","algorithm","inner_fold","trial_batch"}:
            raise ValueError("Unknown parallel execution level")
        if type(self.allow_batched_optuna) is not bool:
            raise ValueError("allow_batched_optuna must be boolean")
        for name in ("max_cpu_fraction", "max_ram_fraction", "max_vram_fraction"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        if (type(self.minimum_workers) is not int or self.minimum_workers < 1
                or type(self.max_parallel_tasks) is not int or self.max_parallel_tasks < 1
                or not math.isfinite(self.monitor_interval_seconds) or self.monitor_interval_seconds <= 0):
            raise ValueError("Resource worker counts and monitor interval must be positive.")

    @classmethod
    def from_config(cls, payload: dict):
        return cls(**payload.get("resources", {}))


@dataclass(frozen=True)
class Workload:
    tasks: int = 1
    memory_per_task_bytes: int | None = None
    vram_per_task_bytes: int | None = None
    require_gpu: bool = False


@dataclass(frozen=True)
class ResourcePlan:
    cpu_budget: int
    parallel_tasks: int
    threads_per_task: int
    ram_budget_bytes: int | None
    vram_budget_bytes: int | None
    gpu_index: int | None


def plan_workload(host: HostResources, policy: ResourcePolicy, work: Workload) -> ResourcePlan:
    if work.tasks < 1 or any(v is not None and v < 1 for v in
                             (work.memory_per_task_bytes, work.vram_per_task_bytes)):
        raise ValueError("Workload estimates and task counts must be positive.")
    threads = max(1, host.available_threads)
    cpu = min(threads, max(policy.minimum_workers, int(threads * policy.max_cpu_fraction), 1))
    parallel = min(cpu, policy.max_parallel_tasks, work.tasks)
    ram = (int(host.ram_available_bytes * policy.max_ram_fraction)
           if host.ram_available_bytes is not None else None)
    if work.memory_per_task_bytes:
        if ram is None:
            parallel = 1
        else:
            if ram < work.memory_per_task_bytes:
                raise ValueError("Insufficient RAM budget for one workload task.")
            parallel = min(parallel, ram // work.memory_per_task_bytes)

    vram = gpu_index = None
    if work.require_gpu:
        candidates = [gpu for gpu in host.gpus if gpu.get("vram_available_bytes") is not None]
        if not candidates:
            raise ValueError("GPU availability/VRAM budget cannot be established.")
        gpu = max(candidates, key=lambda item: item["vram_available_bytes"])
        gpu_index = gpu.get("index")
        vram = int(gpu["vram_available_bytes"] * policy.max_vram_fraction)
        if work.vram_per_task_bytes and vram < work.vram_per_task_bytes:
            raise ValueError("Insufficient GPU VRAM budget for one workload task.")
        # No concurrent GPU workloads without explicit multi-device scheduling.
        parallel = 1
    return ResourcePlan(cpu, parallel, max(1, cpu // parallel), ram, vram, gpu_index)
