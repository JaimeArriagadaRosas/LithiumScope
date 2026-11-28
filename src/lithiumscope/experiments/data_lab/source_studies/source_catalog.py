"""Explicitly approved retirement and reopening of M1 candidate sources."""
from datetime import datetime, timezone
import json
from pathlib import Path
from uuid import uuid4

from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.paths import RESULTS_DIR
from lithiumscope.datasets.management.sources.research import M1_RESEARCH_SOURCES
from lithiumscope.results.atomic_exports import write_json
from lithiumscope.results.eda_artifacts import verify_eda


def known_keys():
    from .directed_profiles import KEYS
    return tuple(spec.key for spec in M1_RESEARCH_SOURCES) + KEYS


def _root(root):
    return Path(root) if root is not None else RESULTS_DIR / "model_1" / "source_catalog"


def _path(key, root):
    if key not in known_keys():
        raise ValueError("Unknown M1 candidate source")
    return _root(root) / "retired" / (key + ".json")


def retirement(key, *, root=None):
    path = _path(key, root)
    if not path.exists():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    if (record.get("source") != key or record.get("disposition") != "rejected"
            or not record.get("reason") or not record.get("decision_reference")
            or not record.get("source_sha256") or not record.get("profile_manifest_sha256")
            or not record.get("all_uses_closed")):
        raise ValueError("Invalid retirement record; explicit review required")
    return record


def retire(key, profile_path, *, reason, decision_reference, all_uses_closed=False, root=None):
    """Never retire from a missing file, failed analysis or an unapproved suggestion."""
    path = _path(key, root)
    if not all_uses_closed or not reason.strip() or not decision_reference.strip():
        raise ValueError("Explicit decision, evidence and closure of all uses required")
    if path.exists():
        raise ValueError("Source already retired")
    profile_path = Path(profile_path)
    manifest_path = profile_path / "manifest.json"
    digest = file_sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("source_key") != key or manifest.get("all_originals_available") is not True
            or not manifest.get("base_sha256") or not manifest.get("source_sha256")
            or not verify_eda(profile_path, manifest.get("signature", ""))
            or file_sha256(manifest_path) != digest):
        raise ValueError("A complete source profile with historical comparison is required")
    record = {"schema_version": 1, "source": key, "disposition": "rejected",
              "all_uses_closed": True, "reason": reason.strip(),
              "decision_reference": decision_reference.strip(),
              "date_utc": datetime.now(timezone.utc).isoformat(),
              "source_sha256": manifest["source_sha256"],
              "profile_manifest_sha256": digest, "profile_path": str(profile_path),
              "profile": manifest}
    write_json(path, record)
    return path


def reopen(key, *, reason, root=None):
    if not reason.strip():
        raise ValueError("Explicit reopening reason required")
    record = retirement(key, root=root)
    if record is None:
        raise ValueError("Source is not retired")
    history = _root(root) / "history" / key / (uuid4().hex + ".json")
    write_json(history, {**record, "reopened_utc": datetime.now(timezone.utc).isoformat(),
                        "reopening_reason": reason.strip()})
    _path(key, root).unlink()
    return history
