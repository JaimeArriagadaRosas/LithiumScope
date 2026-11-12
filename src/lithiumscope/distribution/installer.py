from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import tempfile
import zipfile

import joblib

from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.paths import MODELS_DIR
from lithiumscope.distribution.github_releases import ModelRelease
from lithiumscope.persistence.active_models import set_active_models


@dataclass(frozen=True)
class InstalledModel:
    model_group: str
    run_id: str
    algorithm: str
    model_path: Path
    model_sha256: str


@dataclass(frozen=True)
class InstallationResult:
    release_tag: str
    asset_name: str
    asset_sha256: str
    models: tuple[InstalledModel, ...]
    receipt_path: Path


def _safe_extract(
    archive: Path,
    destination: Path,
    *,
    max_files: int = 500,
    max_uncompressed_bytes: int = 4 * 1024 * 1024 * 1024,
) -> None:
    destination_resolved = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        members = bundle.infolist()
        if len(members) > max_files:
            raise RuntimeError(
                "El bundle contiene demasiados archivos."
            )
        total_uncompressed = sum(
            int(member.file_size)
            for member in members
        )
        if total_uncompressed > max_uncompressed_bytes:
            raise RuntimeError(
                "El bundle excede el tamaño descomprimido permitido."
            )

        for member in members:
            member_path = (
                destination / member.filename
            ).resolve()
            try:
                member_path.relative_to(
                    destination_resolved
                )
            except ValueError as exc:
                raise RuntimeError(
                    "El ZIP contiene una ruta insegura: "
                    f"{member.filename}"
                ) from exc
        bundle.extractall(destination)


def _read_json(path: Path) -> dict:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"JSON inválido dentro del bundle: {path.name}"
        ) from exc
    if not isinstance(payload, dict):
        raise RuntimeError(
            f"JSON inválido dentro del bundle: {path.name}"
        )
    return payload


def _release_manifest(
    root: Path,
    release: ModelRelease,
) -> dict:
    candidates = [
        root / "release_manifest.json",
        *sorted(root.glob("*.json")),
    ]
    seen: set[Path] = set()
    for path in candidates:
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        payload = _read_json(path)
        if not {
            "suggested_tag",
            "model_1",
            "model_2",
        } <= set(payload):
            continue
        if str(payload["suggested_tag"]) != release.tag_name:
            raise RuntimeError(
                "El manifest pertenece a otro tag: "
                f"{payload['suggested_tag']}"
            )
        return payload

    raise RuntimeError(
        "El bundle no contiene un release manifest reconocible."
    )


def _discover_artifacts(
    root: Path,
) -> dict[str, tuple[Path, dict]]:
    discovered: dict[str, tuple[Path, dict]] = {}
    for metadata_path in root.rglob("metadata.json"):
        artifact_dir = metadata_path.parent
        model_path = artifact_dir / "model.joblib"
        if not model_path.is_file():
            continue

        metadata = _read_json(metadata_path)
        model_group = str(
            metadata.get("model_group", "")
        )
        if model_group not in {
            "model_1",
            "model_2",
        }:
            continue
        if model_group in discovered:
            raise RuntimeError(
                "El bundle contiene más de un artefacto para "
                f"{model_group}."
            )

        expected_hash = str(
            metadata.get("model_sha256", "")
        )
        actual_hash = file_sha256(model_path)
        if not expected_hash:
            raise RuntimeError(
                f"{metadata_path} no contiene model_sha256."
            )
        if actual_hash.lower() != expected_hash.lower():
            raise RuntimeError(
                f"Hash inválido para {model_group}: "
                f"esperado={expected_hash} actual={actual_hash}"
            )

        discovered[model_group] = (
            artifact_dir,
            metadata,
        )

    missing = {
        "model_1",
        "model_2",
    } - set(discovered)
    if missing:
        raise RuntimeError(
            "El bundle no contiene todos los modelos requeridos: "
            + ", ".join(sorted(missing))
        )
    return discovered


def _validate_manifest_against_artifacts(
    manifest: dict,
    artifacts: dict[str, tuple[Path, dict]],
) -> None:
    for model_group in ("model_1", "model_2"):
        manifest_model = manifest.get(model_group)
        if not isinstance(manifest_model, dict):
            raise RuntimeError(
                f"Manifest inválido para {model_group}."
            )
        metadata = artifacts[model_group][1]
        expected_run = str(
            manifest_model.get("run_id", "")
        )
        actual_run = str(
            metadata.get("run_id", "")
        )
        if not expected_run or expected_run != actual_run:
            raise RuntimeError(
                f"El run_id de {model_group} no coincide "
                "entre manifest y metadata."
            )


def _validate_deserialization(
    artifacts: dict[str, tuple[Path, dict]],
) -> None:
    for model_group, (artifact_dir, _) in artifacts.items():
        try:
            payload = joblib.load(
                artifact_dir / "model.joblib"
            )
        except Exception as exc:
            raise RuntimeError(
                f"No fue posible cargar {model_group} en este entorno. "
                'Instale las dependencias completas con '
                'pip install -e ".[ml,imagery]" y vuelva a intentarlo.'
            ) from exc
        if not isinstance(payload, dict):
            raise RuntimeError(
                f"El artefacto {model_group} no tiene el formato esperado."
            )


from lithiumscope.distribution.installer_prepare_installation import _prepare_installation


from lithiumscope.distribution.installer_install_release_bundle import install_release_bundle
