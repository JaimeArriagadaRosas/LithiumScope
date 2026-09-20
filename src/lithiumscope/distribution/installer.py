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


def _prepare_installation(
    model_group: str,
    artifact_dir: Path,
    metadata: dict,
) -> tuple[InstalledModel, Path | None, Path]:
    run_id = str(
        metadata.get("run_id")
        or artifact_dir.name
    )
    if not run_id:
        raise RuntimeError(
            f"El artefacto {model_group} no contiene run_id."
        )

    target_root = (
        MODELS_DIR
        / model_group
        / "trained"
    )
    target_root.mkdir(
        parents=True,
        exist_ok=True,
    )
    destination = target_root / run_id
    source_model = artifact_dir / "model.joblib"
    source_hash = file_sha256(source_model)

    if destination.exists():
        existing_model = destination / "model.joblib"
        if (
            existing_model.is_file()
            and file_sha256(existing_model) == source_hash
        ):
            installed = InstalledModel(
                model_group=model_group,
                run_id=run_id,
                algorithm=str(
                    metadata.get(
                        "algorithm",
                        "unknown",
                    )
                ),
                model_path=existing_model,
                model_sha256=source_hash,
            )
            return installed, None, destination
        raise RuntimeError(
            "Ya existe un artefacto distinto con el mismo run_id: "
            f"{destination}"
        )

    staging = (
        target_root
        / f".{run_id}.installing"
    )
    if staging.exists():
        shutil.rmtree(staging)
    shutil.copytree(
        artifact_dir,
        staging,
    )
    staged_model = staging / "model.joblib"
    if (
        not staged_model.is_file()
        or file_sha256(staged_model)
        != source_hash
    ):
        shutil.rmtree(
            staging,
            ignore_errors=True,
        )
        raise RuntimeError(
            f"El modelo {model_group} cambió durante la instalación."
        )

    installed = InstalledModel(
        model_group=model_group,
        run_id=run_id,
        algorithm=str(
            metadata.get(
                "algorithm",
                "unknown",
            )
        ),
        model_path=destination / "model.joblib",
        model_sha256=source_hash,
    )
    return installed, staging, destination


def install_release_bundle(
    archive: Path,
    release: ModelRelease,
    *,
    asset_sha256: str,
) -> InstallationResult:
    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with tempfile.TemporaryDirectory(
        prefix=".release_extract_",
        dir=MODELS_DIR,
    ) as temporary:
        extracted = Path(temporary)
        _safe_extract(
            archive,
            extracted,
        )
        manifest = _release_manifest(
            extracted,
            release,
        )
        artifacts = _discover_artifacts(
            extracted
        )
        _validate_manifest_against_artifacts(
            manifest,
            artifacts,
        )
        _validate_deserialization(
            artifacts
        )

        prepared = [
            _prepare_installation(
                model_group,
                artifact_dir,
                metadata,
            )
            for model_group, (
                artifact_dir,
                metadata,
            ) in sorted(
                artifacts.items()
            )
        ]

        committed: list[Path] = []
        try:
            for _, staging, destination in prepared:
                if staging is None:
                    continue
                os.replace(
                    staging,
                    destination,
                )
                committed.append(
                    destination
                )
        except Exception:
            for destination in reversed(
                committed
            ):
                shutil.rmtree(
                    destination,
                    ignore_errors=True,
                )
            for _, staging, _ in prepared:
                if (
                    staging is not None
                    and staging.exists()
                ):
                    shutil.rmtree(
                        staging,
                        ignore_errors=True,
                    )
            raise

    installed = tuple(
        item
        for item, _, _ in prepared
    )
    active_paths = {
        item.model_group: item.model_path
        for item in installed
    }
    set_active_models(
        active_paths,
        source={
            "type": "github_release",
            "tag": release.tag_name,
            "asset": release.asset_name,
            "release_url": release.html_url,
            "asset_sha256": asset_sha256,
        },
    )

    receipt = {
        "schema_version": 1,
        "release_tag": release.tag_name,
        "release_name": release.release_name,
        "release_url": release.html_url,
        "asset_name": release.asset_name,
        "asset_sha256": asset_sha256,
        "installed_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "models": {
            item.model_group: {
                "run_id": item.run_id,
                "algorithm": item.algorithm,
                "model_path": str(
                    item.model_path
                ),
                "model_sha256": (
                    item.model_sha256
                ),
            }
            for item in installed
        },
    }
    receipt_path = (
        MODELS_DIR
        / "installed_release.json"
    )
    temporary_receipt = (
        receipt_path.with_suffix(
            ".json.tmp"
        )
    )
    temporary_receipt.write_text(
        json.dumps(
            receipt,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    temporary_receipt.replace(
        receipt_path
    )

    return InstallationResult(
        release_tag=release.tag_name,
        asset_name=release.asset_name,
        asset_sha256=asset_sha256,
        models=installed,
        receipt_path=receipt_path,
    )
