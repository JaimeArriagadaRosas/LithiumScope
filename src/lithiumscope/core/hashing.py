from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Iterable


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def directory_sha256(path: Path) -> str:
    digest = sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        relative = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(file_sha256(item).encode("ascii"))
    return digest.hexdigest()


def file_set_sha256(paths: Iterable[Path]) -> str:
    digest = sha256()
    for path in sorted((Path(value) for value in paths), key=lambda item: str(item)):
        digest.update(str(path).encode("utf-8"))
        if path.exists() and path.is_file():
            digest.update(file_sha256(path).encode("ascii"))
        else:
            digest.update(b"MISSING")
    return digest.hexdigest()
