"""Session-relative, integrity-checked prediction artifacts."""
from pathlib import Path
from lithiumscope.core.hashing import file_sha256


def session_relative_path(root: Path, path: Path) -> str:
    return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()


def session_artifact(root: Path, relative: str) -> Path:
    root = Path(root).resolve()
    raw = Path(relative)
    if raw.is_absolute() or ".." in raw.parts or "\\" in relative:
        raise ValueError(f"Invalid session-relative artifact: {relative}")
    path = (root / raw).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Artifact escapes session: {relative}")
    return path


def artifact_hashes(root: Path, artifacts: dict) -> dict[str, str]:
    result = {}
    for name, values in artifacts.items():
        if name == "run_log" or values is None:
            continue
        for relative in values if isinstance(values, list) else [values]:
            path = session_artifact(root, relative)
            if not path.is_file():
                raise FileNotFoundError(path)
            result[relative] = file_sha256(path)
    return result


def verify_report_sources(root: Path, manifest: dict) -> None:
    hashes = manifest.get("artifact_sha256")
    if not hashes:
        if manifest.get("schema_version", 1) >= 2:
            raise ValueError("Missing integrity hashes for versioned prediction session")
        return
    for relative, digest in hashes.items():
        if relative in {"report.pdf", "evaluation.xlsx"}:
            continue
        path = session_artifact(root, relative)
        if not path.is_file() or file_sha256(path) != digest:
            raise ValueError(f"Saved prediction evidence differs: {relative}")
