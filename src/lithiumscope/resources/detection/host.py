"""Read-only host inventory. Unknown hardware properties remain unknown."""
from __future__ import annotations

import csv
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess


@dataclass(frozen=True)
class HostResources:
    logical_cpus: int
    physical_cores: int | None
    available_threads: int
    ram_total_bytes: int | None
    ram_available_bytes: int | None
    disk_free_bytes: int | None
    gpus: tuple[dict, ...]


def detect_gpus() -> tuple[dict, ...]:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return ()
    try:
        result = subprocess.run(
            [executable, "--query-gpu=index,name,memory.total,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=4, check=False,
        )
        if result.returncode:
            return ()
    except (OSError, subprocess.SubprocessError):
        return ()

    def number(value: str, scale: int = 1) -> int | None:
        try:
            return int(float(value.strip()) * scale)
        except (ValueError, OverflowError):
            return None

    devices = []
    for row in csv.reader(result.stdout.splitlines(), skipinitialspace=True):
        if len(row) != 5:
            continue
        index, name, total, free, utilization = row
        devices.append({"index": number(index), "name": name.strip(),
                        "vram_total_bytes": number(total, 1024**2),
                        "vram_available_bytes": number(free, 1024**2),
                        "utilization_percent": number(utilization)})
    return tuple(devices)


def detect_host(root: Path | None = None) -> HostResources:
    logical = max(1, os.cpu_count() or 1)
    available = logical
    physical = total_ram = free_ram = disk = None
    try:
        available = min(available, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        pass
    try:
        import psutil
        physical = psutil.cpu_count(logical=False)
        memory = psutil.virtual_memory()
        total_ram, free_ram = int(memory.total), int(memory.available)
        try:
            available = min(available, len(psutil.Process().cpu_affinity()))
        except (AttributeError, OSError, psutil.Error):
            pass
    except (ImportError, OSError):
        pass
    try:
        disk = shutil.disk_usage(root or Path.cwd()).free
    except OSError:
        pass
    return HostResources(logical, physical, max(1, available), total_ram, free_ram, disk, detect_gpus())
