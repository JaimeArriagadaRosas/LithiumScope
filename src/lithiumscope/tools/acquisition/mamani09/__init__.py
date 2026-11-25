"""Registered Mamani09 public mirror; no claim of official dataset equivalence."""
from pathlib import Path


def acquire_mamani09() -> Path:
    from lithiumscope.tools.acquisition.tabular import ensure_dataset
    return ensure_dataset("mamani09_public_mirror")
