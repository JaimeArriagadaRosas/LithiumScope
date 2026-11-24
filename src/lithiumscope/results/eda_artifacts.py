"""Integrity-checked EDA artifacts; descriptive evidence is not a training holdout."""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from uuid import uuid4

import pandas as pd

from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.reproducibility import canonical_json_hash


def frame_digest(frame: pd.DataFrame) -> str:
    return sha256(frame.to_csv(index=True).encode("utf-8")).hexdigest()


def signature(group: str, source: Path, config: dict, frame: pd.DataFrame) -> str:
    package = Path(__file__).resolve().parents[1]
    roots = [package / group, package / "core" / "scientific", package / "datasets",
             package / "results"]
    code = {p.relative_to(package).as_posix(): file_sha256(p)
            for root in roots for p in sorted(root.rglob("*.py"))}
    return canonical_json_hash({
        "model": group, "scope": "descriptive_full_dataset",
        "source_sha256": file_sha256(source), "config": config,
        "prepared_sha256": frame_digest(frame), "code": code,
    })


def verify_eda(directory: Path, expected_signature: str) -> bool:
    try:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        artifacts = manifest["artifacts"]
        if manifest["signature"] != expected_signature or manifest["state"] != "completed" or not artifacts:
            return False
        actual = {p.relative_to(directory).as_posix() for p in directory.rglob("*")
                  if p.is_file() and p != directory / "manifest.json"}
        if actual != set(artifacts):
            return False
        return all((directory / relative).resolve().is_relative_to(directory.resolve())
                   and file_sha256(directory / relative) == digest
                   for relative, digest in artifacts.items())
    except (OSError, ValueError, KeyError, TypeError):
        return False


def find_compatible(root: Path, expected_signature: str) -> Path | None:
    for path in sorted(root.glob("eda_*"), reverse=True):
        if verify_eda(path, expected_signature):
            return path
    return None


def new_run(root: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    directory = root / f"eda_{stamp}_{uuid4().hex[:8]}"
    (directory / "tables").mkdir(parents=True, exist_ok=False)
    (directory / "figures").mkdir()
    return directory


def complete_run(directory: Path, *, signature_value: str, details: dict) -> Path:
    artifacts = {path.relative_to(directory).as_posix(): file_sha256(path)
                 for path in sorted(directory.rglob("*")) if path.is_file()}
    manifest = {
        "schema_version": 1, "run_id": directory.name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "state": "completed", "scope": "descriptive_full_dataset",
        "training_reusable": False, "signature": signature_value,
        **details, "artifacts": artifacts,
    }
    temporary = directory / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False),
                         encoding="utf-8")
    temporary.replace(directory / "manifest.json")
    return directory


def fail_run(directory: Path, error: Exception) -> None:
    (directory / "failure.json").write_text(json.dumps({
        "state": "failed", "error_type": type(error).__name__, "error": str(error),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
