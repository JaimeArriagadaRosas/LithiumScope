from __future__ import annotations

from importlib import metadata
import json
import os
import platform
import random
import subprocess
import sys
from typing import Any

import numpy as np

from lithiumscope.core.resources import resource_snapshot

_TRACKED_PACKAGES = (
    "numpy",
    "pandas",
    "scikit-learn",
    "matplotlib",
    "xgboost",
    "catboost",
    "optuna",
    "torch",
    "pytorch-tabnet",
    "rasterio",
    "pystac-client",
    "reportlab",
)


def set_global_seed(seed: int, deterministic_torch: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.use_deterministic_algorithms(True, warn_only=True)


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in _TRACKED_PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def runtime_fingerprint() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "hostname": platform.node(),
        "pid": os.getpid(),
        "git_commit": _git_commit(),
        "packages": _package_versions(),
        "resources": resource_snapshot(),
    }


def canonical_json_hash(payload: dict[str, Any]) -> str:
    from hashlib import sha256

    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return sha256(raw.encode("utf-8")).hexdigest()
