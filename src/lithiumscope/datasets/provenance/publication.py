"""Publish staged data and evidence; roll back on ordinary write errors.

The evidence is the commit marker. Readers should verify its data digest.
This is not a filesystem-wide atomic transaction or a cross-process lock.
"""
from __future__ import annotations

from pathlib import Path
from shutil import copy2
from uuid import uuid4


def publish_pair(data_stage: Path, data: Path, evidence_stage: Path, evidence: Path) -> None:
    paths = [Path(p) for p in (data_stage, data, evidence_stage, evidence)]
    if len({p.resolve() for p in paths}) != 4:
        raise ValueError("Staging and destination paths must be distinct")
    staged = [(paths[0], paths[1]), (paths[2], paths[3])]
    backups = {}
    replaced = []
    try:
        for source, dest in staged:
            if not source.is_file() or not source.stat().st_size:
                raise ValueError(f"Missing staged dataset artifact: {source}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                backup = dest.with_name(f".{dest.name}.{uuid4().hex}.backup")
                copy2(dest, backup)
                backups[dest] = backup
        for source, dest in staged:
            source.replace(dest)
            replaced.append(dest)
    except BaseException:
        for dest in reversed(replaced):
            if dest in backups:
                backups[dest].replace(dest)
            else:
                dest.unlink(missing_ok=True)
        raise
    finally:
        for backup in backups.values():
            backup.unlink(missing_ok=True)
