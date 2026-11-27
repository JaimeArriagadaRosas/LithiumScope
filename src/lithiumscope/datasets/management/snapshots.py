"""Content-addressed experimental snapshots and reusable source transformations."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
import hashlib
import json

from lithiumscope.core.hashing import file_sha256


def _digest(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Snapshot:
    directory: Path
    version_id: str
    reused: bool

    @property
    def data_path(self) -> Path:
        return self.directory / "dataset.csv"


def _identity(sources: dict[str, Path], policy: dict, transform_revision: str) -> dict:
    if not sources or not transform_revision.strip():
        raise ValueError("Snapshot requires named inputs and a transformation revision")
    if any(not isinstance(name, str) or not name.strip() for name in sources):
        raise ValueError("Empty source identity")
    return {"schema_version": 1, "sources": {name: file_sha256(path) for name, path in sorted(sources.items())},
            "policy": json.loads(json.dumps(policy, allow_nan=False)), "transform_revision": transform_revision}


def verify_snapshot(directory: Path, expected_identity: dict | None = None) -> dict:
    directory = Path(directory)
    metadata = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    identity = metadata["identity"]
    if metadata["version_id"] != _digest(identity) or directory.name != metadata["version_id"]:
        raise ValueError("Snapshot identity mismatch")
    if expected_identity is not None and identity != expected_identity:
        raise ValueError("Snapshot input contract mismatch")
    data = directory / "dataset.csv"
    if not data.is_file() or not data.stat().st_size or file_sha256(data) != metadata["data_sha256"]:
        raise ValueError("Snapshot data integrity mismatch")
    artifacts = metadata.get("artifacts", {"dataset.csv": metadata["data_sha256"]})
    if artifacts.get("dataset.csv") != metadata["data_sha256"]:
        raise ValueError("Snapshot data digest mismatch")
    actual = {p.name for p in directory.iterdir() if p.name != "manifest.json"}
    if actual != set(artifacts):
        raise ValueError("Snapshot artifact inventory mismatch")
    for name, digest in artifacts.items():
        path = directory / name
        if Path(name).name != name or path.is_symlink() or not path.is_file() or file_sha256(path) != digest:
            raise ValueError(f"Snapshot artifact integrity mismatch: {name}")
    return metadata


def materialize_snapshot(root: Path, sources: dict[str, Path], *, policy: dict,
                         transform_revision: str, transform) -> Snapshot:
    """Build an immutable candidate artifact; reuse only verified matching content.

    transform receives a copy of the named input paths and a staged CSV output.
    Caller declares the transformation revision; changed code/policy must use a new revision.
    No frozen path or production configuration is updated.
    """
    root = Path(root)
    sources = {key: Path(path).resolve() for key, path in sources.items()}
    identity = _identity(sources, policy, transform_revision)
    version = _digest(identity)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / version
    if destination.exists():
        verify_snapshot(destination, identity)
        return Snapshot(destination, version, True)
    with TemporaryDirectory(prefix=".build-", dir=root) as staging:
        folder = Path(staging) / version
        folder.mkdir()
        data = folder / "dataset.csv"
        transform(dict(sources), data)
        if not data.is_file() or not data.stat().st_size:
            raise ValueError("Transformation did not produce a dataset")
        if _identity(sources, policy, transform_revision) != identity:
            raise ValueError("Sources or transformation policy changed during snapshot construction")
        artifacts = {}
        for path in folder.iterdir():
            if path.name == "manifest.json" or path.is_symlink() or not path.is_file():
                raise ValueError(f"Unexpected staged artifact: {path.name}")
            artifacts[path.name] = file_sha256(path)
        metadata = {"identity": identity, "version_id": version, "data_sha256": artifacts["dataset.csv"],
                    "artifacts": artifacts, "scientific_quality_certified": False}
        (folder / "manifest.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        try:
            folder.rename(destination)
        except OSError:
            # A concurrent publisher may have won; never overwrite a snapshot.
            if not destination.exists():
                raise
            verify_snapshot(destination, identity)
            return Snapshot(destination, version, True)
    verify_snapshot(destination, identity)
    from lithiumscope.core.logging.session import get_logger
    get_logger("datasets.snapshots").info("Candidate snapshot published: %s", version)
    return Snapshot(destination, version, False)


def materialize_sources(root: Path, sources: dict[str, Path], *, policy: dict,
                        transform_revision: str, transform) -> dict[str, Snapshot]:
    """Cache independent per-source transformations; changed sources alone are rebuilt.

    Cross-source reconciliation must be performed after this stage.
    """
    return {name: materialize_snapshot(root, {name: path}, policy=policy,
                                      transform_revision=transform_revision, transform=transform)
            for name, path in sorted(sources.items())}
