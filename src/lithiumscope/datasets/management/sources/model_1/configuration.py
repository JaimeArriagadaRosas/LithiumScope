"""Read-only path and frozen snapshot configuration."""
from pathlib import Path
from lithiumscope.core.hashing import file_sha256


def project_path(raw, root):
    path = Path(raw).expanduser()
    return (path if path.is_absolute() else Path(root) / path).resolve()


def verified_frozen_source(path, expected, *, hash_file=file_sha256):
    if not path.is_file():
        raise FileNotFoundError("Frozen M1 dataset unavailable; restore verified bytes or explicitly select a separate unfrozen candidate configuration")
    if hash_file(path) != expected:
        raise ValueError("Frozen combined M1 dataset hash mismatch; no automatic overwrite")
    return path
