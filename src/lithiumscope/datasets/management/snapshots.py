"""Content-addressed experimental snapshots and reusable source transformations."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
import hashlib
import json
import platform
from importlib.metadata import version, PackageNotFoundError
from uuid import uuid4

from lithiumscope.core.logging.session import get_logger

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


def dataset_implementation_revision(label: str, *, dataset_root: Path | None = None) -> str:
    """Conservative content fingerprint; independent of checkout location and Git branch."""
    root = Path(dataset_root) if dataset_root is not None else Path(__file__).resolve().parents[1]
    modules = {path.relative_to(root).as_posix(): file_sha256(path)
               for path in sorted(root.rglob("*.py")) if "__pycache__" not in path.parts}
    if not modules:
        raise ValueError("Dataset implementation unavailable")
    packages = {}
    for name in ("numpy", "pandas", "openpyxl"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "missing"
    return label + ":" + _digest({"modules": modules, "packages": packages, "python": platform.python_version()})


def _identity(sources: dict[str, Path], policy: dict, transform_revision: str) -> dict:
    if not sources or not transform_revision.strip():
        raise ValueError("Snapshot requires named inputs and a transformation revision")
    if any(not isinstance(name, str) or not name.strip() for name in sources):
        raise ValueError("Empty source identity")
    return {"schema_version": 1, "sources": {name: file_sha256(path) for name, path in sorted(sources.items())},
            "policy": json.loads(json.dumps(policy, allow_nan=False)), "transform_revision": transform_revision}


def verify_snapshot(directory: Path, expected_identity: dict | None = None) -> dict:
    directory = Path(directory)
    manifest = directory / "manifest.json"
    if directory.is_symlink() or manifest.is_symlink():
        raise ValueError("Snapshot may not be a symbolic link")
    metadata = json.loads(manifest.read_text(encoding="utf-8"))
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
                         transform_revision: str, transform, recover_corrupt: bool = False) -> Snapshot:
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
    logger = get_logger("datasets.snapshots")
    if destination.exists() or destination.is_symlink():
        try:
            verify_snapshot(destination, identity)
        except (OSError, ValueError, KeyError, TypeError):
            logger.warning("Candidate snapshot integrity failure: %s", version)
            if not recover_corrupt or destination.is_symlink() or not destination.is_dir():
                raise
            quarantine = root / ".quarantine"
            if quarantine.is_symlink():
                raise ValueError("Snapshot quarantine may not be a symbolic link")
            quarantine.mkdir(exist_ok=True)
            damaged = quarantine / f"{version}.{uuid4().hex}"
            destination.rename(damaged)
            logger.warning("Corrupted candidate preserved for explicit recovery: %s", damaged)
        else:
            logger.info("Candidate snapshot reused: %s", version)
            return Snapshot(destination, version, True)
    logger.info("Building candidate snapshot: %s", version)
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
    logger.info("Candidate snapshot published: %s", version)
    return Snapshot(destination, version, False)


def materialize_sources(root: Path, sources: dict[str, Path], *, policy: dict,
                        transform_revision: str, transform, recover_corrupt: bool = False) -> dict[str, Snapshot]:
    """Cache independent per-source transformations; changed sources alone are rebuilt.

    Cross-source reconciliation must be performed after this stage.
    """
    return {name: materialize_snapshot(root, {name: path}, policy=policy,
                                      transform_revision=transform_revision, transform=transform, recover_corrupt=recover_corrupt)
            for name, path in sorted(sources.items())}


def list_snapshots(root: Path) -> list[dict]:
    """Read-only inventory of version directories, including incomplete/corrupt versions."""
    import re
    root = Path(root).resolve()
    records = []
    for directory in sorted(root.rglob("*")):
        relative = directory.relative_to(root)
        if any(part.startswith(".") for part in relative.parts) or not directory.is_dir():
            continue
        if not re.fullmatch(r"[0-9a-f]{64}", directory.name):
            continue
        try:
            if directory.is_symlink() or not directory.resolve().is_relative_to(root):
                raise ValueError("Snapshot directory escapes catalog")
            metadata = verify_snapshot(directory)
            records.append({"path": relative.as_posix(), "version_id": directory.name, "state": "verified",
                            "identity": metadata["identity"], "artifacts": metadata.get("artifacts", {})})
        except (OSError, ValueError, KeyError, TypeError) as exc:
            records.append({"path": relative.as_posix(), "version_id": directory.name, "state": "corrupted",
                            "reason": str(exc)})
    return records
