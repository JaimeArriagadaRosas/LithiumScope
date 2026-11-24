"""Read-only discovery of local and versioned result evidence."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from zipfile import BadZipFile, ZipFile

from lithiumscope.core.hashing import file_sha256
from lithiumscope.persistence.active_models import active_model_path


def read_record(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _inside(path: Path, root: Path) -> bool:
    try:
        return path.is_file() and path.resolve().is_relative_to(root.resolve())
    except OSError:
        return False


def _timestamp(path: Path, record: dict | None = None) -> float:
    for key in ("completed_at_utc", "created_at_utc", "saved_at_utc", "updated_at_utc"):
        raw = (record or {}).get(key)
        if raw:
            try:
                value = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
                return value.replace(tzinfo=value.tzinfo or timezone.utc).timestamp()
            except (ValueError, OverflowError):
                continue
    return path.stat().st_mtime


def latest_record(root: Path, name: str) -> tuple[Path, dict] | None:
    candidates = []
    for path in root.rglob(name):
        if _inside(path, root):
            record = read_record(path)
            if record:
                try:
                    candidates.append((_timestamp(path, record), str(path), path, record))
                except OSError:
                    continue
    if not candidates:
        return None
    _, _, path, record = max(candidates, key=lambda item: item[:2])
    return path, record


def _document_valid(path: Path) -> bool:
    try:
        if path.suffix.lower() == ".pdf":
            # Structural screening, not a full PDF integrity certification.
            with path.open("rb") as stream:
                if not stream.read(8).startswith(b"%PDF-"):
                    return False
                stream.seek(max(0, path.stat().st_size - 2048))
                tail = stream.read()
            return b"startxref" in tail and tail.rstrip().endswith(b"%%EOF")
        if path.suffix.lower() == ".xlsx":
            with ZipFile(path) as archive:
                if archive.testzip() is not None:
                    return False
                if not {"[Content_Types].xml", "xl/workbook.xml"}.issubset(archive.namelist()):
                    return False
            from openpyxl import load_workbook
            workbook = load_workbook(path, read_only=True)
            try:
                return bool(workbook.sheetnames)
            finally:
                workbook.close()
    except (OSError, ValueError, BadZipFile, KeyError):
        return False
    except Exception:
        # Corrupt XML and unsupported workbook formats are not evidence.
        return False
    return False


def _related_record(path: Path, root: Path) -> dict:
    for parent in (path.parent, *path.parent.parents):
        if not parent.is_relative_to(root):
            break
        for name in ("prediction_manifest.json", "run.json", "manifest.json"):
            record = read_record(parent / name)
            if record:
                return record
    return {}


def latest_document(root: Path, suffix: str) -> Path | None:
    root = root.resolve()
    candidates = []
    for path in root.rglob("*"):
        if path.suffix.lower() != suffix or not _inside(path, root):
            continue
        if not _document_valid(path):
            continue
        record = _related_record(path, root)
        try:
            candidates.append((_timestamp(path, record), str(path), path))
        except OSError:
            continue
    return max(candidates, default=(0, "", None), key=lambda item: item[:2])[2]


def document_identity(path: Path, root: Path) -> dict:
    return _related_record(path.resolve(), root.resolve())


def active_summary(group: str) -> dict:
    path = active_model_path(group)
    if path is None:
        return {"model": group, "state": "N/D", "reason": "No hay modelo activo registrado y disponible."}
    metadata = read_record(path.parent / "metadata.json") or {}
    expected = metadata.get("model_sha256")
    try:
        integrity = "verified" if expected and file_sha256(path) == expected else (
            "mismatch" if expected else "N/D"
        )
    except OSError:
        integrity = "N/D"
    return {
        "model": group, "state": "ACTIVO" if integrity != "mismatch" else "ERROR",
        "algorithm": metadata.get("algorithm", "N/D"),
        "run_id": metadata.get("run_id", path.parent.name),
        "metrics": metadata.get("metrics", {}),
        "metric_scope": metadata.get("metric_scope", "N/D"),
        "integrity": integrity, "path": str(path),
    }
