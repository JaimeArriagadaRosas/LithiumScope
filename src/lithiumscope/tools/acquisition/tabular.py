from __future__ import annotations

import json
import hashlib
from uuid import uuid4
from pathlib import Path
from typing import Iterable

import requests

from lithiumscope.core.exceptions import DatasetError
from lithiumscope.core.logging.session import get_logger
from lithiumscope.core.paths import DATA_DIR
from lithiumscope.datasets.management.registry import DatasetSpec, get_dataset_spec
from lithiumscope.datasets.validation.files import validate_nonempty
from lithiumscope.runtime.console_status import Spinner

logger = get_logger("datasets")


def _stream_download(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.part")
    spinner = Spinner(f"Descargando dataset: {destination.name}").start()
    logger.info("Downloading %s -> %s", url, destination)

    try:
        with requests.get(url, stream=True, timeout=60) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length", "0") or 0)
            if response.headers.get("content-encoding", "identity").lower() != "identity":
                total = 0  # iter_content decodes transport compression
            downloaded = 0
            with temporary.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    handle.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        spinner.update(
                            f"Descargando {destination.name}: "
                            f"{downloaded / 1048576:.1f}/"
                            f"{total / 1048576:.1f} MiB"
                        )
                    else:
                        spinner.update(
                            f"Descargando {destination.name}: "
                            f"{downloaded / 1048576:.1f} MiB"
                        )
        if downloaded == 0 or (total and downloaded != total):
            raise DatasetError("Incomplete dataset download")
        temporary.replace(destination)
        spinner.succeed(f"Dataset descargado: {destination.name}")
        return validate_nonempty(destination)
    except Exception:
        spinner.fail(f"Falló la descarga: {destination.name}")
        temporary.unlink(missing_ok=True)
        raise


def _zenodo_manifest(record: int) -> list[dict]:
    response = requests.get(
        f"https://zenodo.org/api/records/{record}",
        timeout=60,
    )
    response.raise_for_status()
    files = response.json().get("files", [])
    if not files:
        raise DatasetError(
            f"Zenodo record {record} contains no downloadable files."
        )
    return files


def _selected_zenodo_files(spec: DatasetSpec, manifest: list[dict]) -> list[dict]:
    selected = set(spec.zenodo_files)
    records = {}
    for item in manifest:
        key = item.get("key") or item.get("filename")
        if not key or (selected and key not in selected):
            continue
        if key in records or Path(key).name != key or "/" in key or chr(92) in key or ":" in key or key in {".", "..", "zenodo_manifest.json"}:
            raise DatasetError(f"Unsafe or duplicate Zenodo filename: {key}")
        checksum = str(item.get("checksum", "")).split(":", 1)
        if len(checksum) != 2 or checksum[0] not in {"md5", "sha256"} or len(checksum[1]) != (32 if checksum[0] == "md5" else 64) or any(c not in "0123456789abcdef" for c in checksum[1].lower()):
            raise DatasetError(f"Missing or invalid Zenodo checksum: {key}")
        if not isinstance(item.get("size"), int) or item["size"] <= 0:
            raise DatasetError(f"Missing Zenodo file size: {key}")
        records[key] = {**item, "key": key}
    if not records or selected - set(records):
        raise DatasetError(f"Zenodo selection incomplete: {spec.key}; missing={sorted(selected-set(records))}")
    return list(records.values())


def _verify_zenodo_file(path: Path, item: dict) -> bool:
    if path.is_symlink() or not path.is_file() or path.stat().st_size != item["size"]:
        return False
    algorithm, expected = item["checksum"].split(":", 1)
    digest = hashlib.new(algorithm, usedforsecurity=False)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest() == expected.lower()


def _zenodo_download(spec: DatasetSpec, destination_dir: Path) -> Path:
    if spec.zenodo_record is None:
        raise DatasetError(f"Zenodo record missing for {spec.key}")
    if destination_dir.is_symlink():
        raise DatasetError("Zenodo destination may not be a symbolic link")
    destination_dir.mkdir(parents=True, exist_ok=True)
    evidence = destination_dir / "zenodo_manifest.json"
    # A completed local manifest is usable offline only when every selected byte matches.
    if evidence.is_file() and not evidence.is_symlink():
        try:
            cached = json.loads(evidence.read_text(encoding="utf-8"))
            if cached.get("record") == spec.zenodo_record:
                files = _selected_zenodo_files(spec, cached["files"])
                if all(_verify_zenodo_file(destination_dir / item["key"], item) for item in files):
                    logger.info("Verified cached Zenodo record %s files=%d", spec.zenodo_record, len(files))
                    return destination_dir
        except (OSError, ValueError, KeyError, TypeError, AttributeError, DatasetError):
            logger.warning("Invalid cached Zenodo evidence: %s", evidence)
    files = _selected_zenodo_files(spec, _zenodo_manifest(spec.zenodo_record))
    for item in files:
        target = destination_dir / item["key"]
        if _verify_zenodo_file(target, item):
            logger.info("Reusing verified Zenodo file %s", target)
            continue
        links = item.get("links", {})
        url = links.get("content") or links.get("self")
        if not url:
            raise DatasetError(f"No download URL for {item['key']}")
        staged = destination_dir / f".{uuid4().hex}.download"
        try:
            _stream_download(url, staged)
            if not _verify_zenodo_file(staged, item):
                raise DatasetError(f"Zenodo checksum or size mismatch: {item['key']}")
            staged.replace(target)
        finally:
            staged.unlink(missing_ok=True)
    staged = destination_dir / f".manifest.{uuid4().hex}.tmp"
    try:
        staged.write_text(json.dumps({"record": spec.zenodo_record, "files": files}, indent=2), encoding="utf-8")
        staged.replace(evidence)
    finally:
        staged.unlink(missing_ok=True)
    logger.info("Verified Zenodo record %s files=%d", spec.zenodo_record, len(files))
    return destination_dir


def ensure_dataset(key: str, allow_large: bool = False) -> Path:
    spec = get_dataset_spec(key)
    base = DATA_DIR / "raw" / spec.model

    if spec.provider == "manual":
        raise DatasetError(
            f"{spec.key} is a reviewed/manual source and is not "
            "downloaded automatically. Review the source and place "
            "the approved files in the configured raw-data directory: "
            f"{spec.source_url}"
        )

    if spec.provider == "dynamic":
        raise DatasetError(
            f"{spec.key} is a dynamic imagery provider, not a static dataset. "
            "Use its scene acquisition pipeline when coordinates are available."
        )

    if spec.large and not allow_large:
        raise DatasetError(
            f"{spec.key} is marked as a large download. "
            f"Re-run with allow_large=True after reviewing the source: "
            f"{spec.source_url}"
        )

    if spec.provider == "direct":
        destination = base / spec.destination_name
        if destination.exists() and destination.stat().st_size > 0:
            return destination
        if not spec.download_url:
            raise DatasetError(f"No direct URL configured for {spec.key}")
        return _stream_download(spec.download_url, destination)

    if spec.provider == "zenodo":
        destination = base / spec.destination_name
        return _zenodo_download(spec, destination)

    raise DatasetError(f"Unsupported dataset provider: {spec.provider}")


def ensure_many(
    keys: Iterable[str],
    allow_large: bool = False,
) -> list[Path]:
    return [
        ensure_dataset(key, allow_large=allow_large)
        for key in keys
    ]
