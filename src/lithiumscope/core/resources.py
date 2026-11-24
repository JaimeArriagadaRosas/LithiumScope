"""Compatibility imports; canonical resource management lives in resources/."""
from lithiumscope.resources.manager import cpu_worker_budget, resource_snapshot

__all__ = ["cpu_worker_budget", "resource_snapshot"]
