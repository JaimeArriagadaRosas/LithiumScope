"""Bounded evidence inventory: no file contents, credentials or raw sample rows."""
from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET

from lithiumscope.core.hashing import file_sha256


def local_path(root: Path, value: str | Path) -> Path:
    root = root.resolve()
    raw = Path(value)
    path = (raw if raw.is_absolute() else root / raw).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Path outside development workspace")
    return path


def file_record(root: Path, value: str | Path) -> dict:
    try:
        path = local_path(root, value)
    except (ValueError, OSError):
        return {"state": "BLOCKED", "reason": "outside_workspace_or_invalid_path"}
    label = path.relative_to(root.resolve()).as_posix()
    if not path.is_file():
        return {"state": "MISSING", "path": label}
    try:
        size = path.stat().st_size
        return {"state": "OK" if size else "EMPTY", "path": label,
                "bytes": size, "sha256": file_sha256(path)}
    except OSError:
        return {"state": "ERROR", "path": label, "reason": "unreadable"}


def directory_record(root: Path, value: str | Path) -> dict:
    try:
        path = local_path(root, value)
    except (ValueError, OSError):
        return {"state": "BLOCKED", "reason": "outside_workspace_or_invalid_path"}
    label = path.relative_to(root.resolve()).as_posix()
    if not path.is_dir():
        return {"state": "MISSING", "path": label}
    try:
        # Count only direct files and directories, not contents or external targets.
        children = [p for p in path.iterdir() if not p.is_symlink() and p.name != ".gitkeep"]
        return {"state": "OK" if children else "EMPTY", "path": label,
                "direct_files": sum(p.is_file() for p in children),
                "direct_directories": sum(p.is_dir() for p in children)}
    except OSError:
        return {"state": "ERROR", "path": label, "reason": "unreadable"}


def read_local_json(root: Path, value: str | Path) -> dict:
    path = local_path(root, value)
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object")
    return data


def git_identity(root: Path) -> dict:
    def git(*args):
        result = subprocess.run(["git", "-C", str(root), *args], check=True,
                                capture_output=True, text=True, timeout=10)
        return result.stdout.strip()
    try:
        head = git("rev-parse", "HEAD")
        if not re.fullmatch(r"[0-9a-f]{40,64}", head):
            raise ValueError("Invalid commit")
        branch = git("branch", "--show-current")
        dirty = bool(git("status", "--porcelain", "--untracked-files=no"))
        untracked = bool(git("ls-files", "--others", "--exclude-standard", "--",
                             "src", "tests", "scripts", "config", "docs/integral-refactor"))
        return {"state": "WARNING" if dirty or untracked else "OK",
                "commit": head, "branch": branch or "detached",
                "tracked_worktree_dirty": dirty, "untracked_code_present": untracked,
                "untracked_files_inspected": True}
    except (OSError, subprocess.SubprocessError, ValueError):
        return {"state": "UNAVAILABLE"}


def latest_test_report(root: Path, current_commit: str | None) -> dict:
    directory = local_path(root, "results/tests")
    candidates = []
    for path in directory.glob("*/summary.json"):
        try:
            safe = local_path(root, path)
            candidates.append((safe.stat().st_mtime_ns, safe.as_posix(), safe))
        except (ValueError, OSError):
            continue
    if not candidates:
        return {"state": "MISSING", "reason": "No local regression summary; use option 5.2"}
    return test_report(root, max(candidates)[2], current_commit)


def test_report(root: Path, path: Path, current_commit: str | None) -> dict:
    report = read_local_json(root, path)
    counts = report.get("counts", {})
    if not isinstance(counts, dict):
        raise ValueError("Invalid test counts")
    selected = {k: counts.get(k) for k in ("tests", "failures", "errors", "skipped")}
    coverage = None
    coverage_path = path.parent / "coverage.xml"
    if coverage_path.is_file():
        node = ET.parse(local_path(root, coverage_path)).getroot()
        value = float(node.attrib["line-rate"])
        if not 0 <= value <= 1:
            raise ValueError("Invalid coverage rate")
        coverage = 100 * value
    source_commit = report.get("git_commit")
    matches = (source_commit == current_commit and report.get("tracked_worktree_dirty") is False
               and report.get("untracked_code_present") is False
               and report.get("git_identity_unchanged") is True
               if isinstance(source_commit, str) and current_commit else None)
    return {"state": "OK" if report.get("state") == "passed" else "WARNING",
            "test_state": report.get("state"), "counts": selected,
            "duration_seconds": report.get("duration_seconds"),
            "line_coverage_pct": coverage, "commit": source_commit,
            "matches_current_clean_commit": matches,
            "summary": file_record(root, path),
            "junit": file_record(root, path.parent / "junit.xml"),
            "coverage": file_record(root, coverage_path),
            "log": file_record(root, report["log"]) if isinstance(report.get("log"), str) else None}


def active_models(root: Path) -> dict:
    registry = local_path(root, "models/active_models.json")
    entries = read_local_json(root, registry).get("models", {}) if registry.is_file() else {}
    if not isinstance(entries, dict):
        raise ValueError("Invalid active model registry")
    reports = {}
    for group in ("model_1", "model_2"):
        entry = entries.get(group, {})
        value = entry.get("model_path") if isinstance(entry, dict) else None
        if not isinstance(value, str):
            reports[group] = {"state": "NOT_ACTIVE"}
            continue
        bundle = file_record(root, value)
        if bundle["state"] != "OK":
            reports[group] = bundle
            continue
        path = local_path(root, value)
        metadata = path.parent / "metadata.json"
        record = read_local_json(root, metadata) if metadata.is_file() else {}
        expected = record.get("model_sha256")
        reports[group] = {"state": "VERIFIED" if expected == bundle["sha256"] else
                          "MISMATCH" if expected else "UNVERIFIED",
                          "bundle": bundle, "metadata": file_record(root, metadata)}
    return {"state": "OK" if all(v["state"] == "VERIFIED" for v in reports.values()) else "WARNING",
            "models": reports, "deserialized": False}
