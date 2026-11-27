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

def validate_file_integrity(path: Path, *, expected_sha256: str | None = None,
                            root: Path | None = None) -> Path:
    """Validate a nonempty regular file and optional content identity."""
    from lithiumscope.core.hashing import file_sha256
    raw = Path(path).expanduser()
    resolved = ((Path(root) / raw) if root is not None and not raw.is_absolute() else raw).resolve()
    if root is not None and not resolved.is_relative_to(Path(root).resolve()):
        raise DatasetError("Dataset path escapes declared root")
    validate_nonempty(resolved)
    if not resolved.is_file():
        raise DatasetError(f"Expected regular dataset file: {resolved}")
    if expected_sha256 is not None and file_sha256(resolved) != expected_sha256:
        raise DatasetError(f"Dataset hash mismatch: {resolved}")
    return resolved
