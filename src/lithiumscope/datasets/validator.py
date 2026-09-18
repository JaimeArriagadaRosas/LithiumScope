from __future__ import annotations

from pathlib import Path

from lithiumscope.core.exceptions import DatasetError


def validate_nonempty(path: Path) -> Path:
    if not path.exists():
        raise DatasetError(f"Dataset not found: {path}")
    if path.is_file() and path.stat().st_size == 0:
        raise DatasetError(f"Dataset is empty: {path}")
    if path.is_dir() and not any(path.iterdir()):
        raise DatasetError(f"Dataset directory is empty: {path}")
    return path
