"""Resolve registered research originals without acquisition or reconstruction."""
from __future__ import annotations

import json
from pathlib import Path
import re

from lithiumscope.core.hashing import file_sha256
from lithiumscope.datasets.management.sources.research import ResearchPartition


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _valid_name(name):
    return isinstance(name, str) and Path(name).name == name and not any(c in name for c in ("/", chr(92)))


def _status(path, expected):
    if not path.is_file():
        return {"file": path.name, "state": "missing"}
    if path.is_symlink() or not path.resolve().is_relative_to(path.parent.resolve()):
        return {"file": path.name, "state": "blocked", "reason": "unsafe path"}
    try:
        digest = file_sha256(path)
    except OSError as exc:
        return {"file": path.name, "state": "blocked", "reason": str(exc)}
    if expected and digest != expected:
        return {"file": path.name, "state": "blocked", "reason": "hash mismatch", "sha256": digest}
    return {"file": path.name, "state": "verified" if expected else "unverified", "sha256": digest}


def georoc_files(spec, root):
    """Read all declared partitions, reporting missing and corrupt files individually."""
    folder = root / spec.key
    plan_path, receipt = folder / "partition_plan.json", folder / "acquisition.json"
    plan_verified = False
    known = {}
    if plan_path.is_file():
        plan = _read(plan_path)
        if plan["doi"] != spec.doi:
            raise ValueError("Research partition plan DOI differs")
        partitions = tuple(ResearchPartition(**p) for p in plan["partitions"])
        if spec.partitions and partitions != spec.partitions:
            raise ValueError("Pinned research plan differs")
        plan_verified = True
    elif spec.partitions:
        partitions = spec.partitions
        plan_verified = True
    else:
        # Local discovery is diagnostic only: no claim of complete published plan.
        pattern = r"^\d{4}-\d{2}-2JETOA_(" + "|".join(map(re.escape, spec.rock_types)) + r")(?:_part\d+)?\.csv$"
        partitions = tuple(ResearchPartition(i, p.name) for i, p in enumerate(
            sorted(folder.glob("*.csv")), 1) if re.fullmatch(pattern, p.name, re.I))
    if receipt.is_file():
        record = _read(receipt)
        if (record.get("doi") != spec.doi or not plan_path.is_file()
                or record.get("plan_sha256") != file_sha256(plan_path)):
            raise ValueError("Research acquisition receipt differs")
        known = {entry["identifier"]: entry for entry in record["files"]}
    if (len({p.filename for p in partitions}) != len(partitions)
            or len({p.identifier for p in partitions}) != len(partitions)):
        raise ValueError("Duplicate research partitions")
    files, inventory = [], []
    for partition in partitions:
        if not _valid_name(partition.filename) or not partition.filename.lower().endswith(".csv"):
            raise ValueError("Invalid research partition")
        entry = known.get(partition.identifier, {})
        if entry and entry.get("filename") != partition.filename:
            raise ValueError("Acquisition file identity differs")
        path = folder / partition.filename
        status = _status(path, entry.get("sha256"))
        inventory.append(status)
        if status["state"] in ("verified", "unverified"):
            files.append(path)
    if not partitions:
        inventory.append({"file": "partition_plan.json", "state": "missing"})
    return tuple(files), inventory, receipt if receipt.is_file() else None, plan_verified


def directed_files(expected, folder):
    """All named originals are checked, not just the first workbook."""
    paths, inventory = {}, []
    for name, digest in expected.items():
        if not _valid_name(name):
            raise ValueError("Invalid directed source name")
        path = folder / name
        record = _status(path, digest)
        inventory.append(record)
        if record["state"] in ("verified", "unverified"):
            paths[name] = path
    return paths, inventory
