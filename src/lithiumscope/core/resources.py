from __future__ import annotations

import os

from lithiumscope.core.config import load_config


def cpu_worker_budget() -> int:
    config = load_config("app").get("resources", {})
    total = max(1, os.cpu_count() or 1)
    fraction = float(config.get("max_cpu_fraction", 0.75))
    minimum = max(1, int(config.get("minimum_workers", 1)))
    return max(minimum, min(total, int(total * fraction) or 1))


def resource_snapshot() -> dict:
    payload = {
        "cpu_count": os.cpu_count(),
        "cpu_worker_budget": cpu_worker_budget(),
    }
    try:
        import torch

        payload["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            payload["cuda_device"] = torch.cuda.get_device_name(0)
            properties = torch.cuda.get_device_properties(0)
            payload["cuda_total_memory_bytes"] = int(properties.total_memory)
    except ImportError:
        payload["cuda_available"] = False
    return payload
