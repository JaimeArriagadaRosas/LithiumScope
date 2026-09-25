from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import zipfile

from lithiumscope.core.paths import MODELS_DIR, RESULTS_DIR
from lithiumscope.persistence.active_models import (
    active_model_path,
    read_active_models,
)
from lithiumscope.results.catalog import active_release_candidate


def _active_artifact_release_entry(
    model_group: str,
) -> dict | None:
    catalog_entry = active_release_candidate(
        model_group
    )
    if catalog_entry is not None:
        return {
            **catalog_entry,
            "release_provenance": "local_run_catalog",
            "reused_from_previous_release": False,
        }

    model_path = active_model_path(
        model_group
    )
    if model_path is None:
        return None
    metadata_path = (
        model_path.parent / "metadata.json"
    )
    if not metadata_path.is_file():
        return None
    try:
        metadata = json.loads(
            metadata_path.read_text(
                encoding="utf-8"
            )
        )
    except (
        OSError,
        json.JSONDecodeError,
    ):
        return None

    registry = read_active_models()
    registry_entry = (
        registry.get("models", {})
        .get(model_group, {})
    )
    source = (
        registry_entry.get("source", {})
        if isinstance(
            registry_entry,
            dict,
        )
        else {}
    )
    return {
        "model_group": model_group,
        "run_id": str(
            metadata.get(
                "run_id",
                model_path.parent.name,
            )
        ),
        "winner": metadata.get(
            "variant_id",
            metadata.get("algorithm"),
        ),
        "algorithm": metadata.get("algorithm"),
        "dataset_sha256": metadata.get(
            "dataset_sha256"
        ),
        "git_commit": metadata.get(
            "git_commit"
        ),
        "model_sha256": metadata.get(
            "model_sha256"
        ),
        "release_candidate": True,
        "release_provenance": "active_installed_artifact",
        "reused_from_previous_release": (
            source.get("type")
            == "github_release"
        ),
        "source_release": source.get("tag"),
        "source_release_asset": source.get(
            "asset"
        ),
    }


def build_release_candidate_manifest() -> Path:
    model_1 = _active_artifact_release_entry(
        "model_1"
    )
    model_2 = _active_artifact_release_entry(
        "model_2"
    )

    if model_1 is None or model_2 is None:
        missing = []
        if model_1 is None:
            missing.append("model_1")
        if model_2 is None:
            missing.append("model_2")
        raise RuntimeError(
            "No existe un modelo activo con artefacto verificable en: "
            + ", ".join(missing)
        )

    stamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")
    suggested_tag = (
        "lithiumscope-"
        + str(model_1["run_id"])
    )
    payload = {
        "schema_version": 2,
        "artifact_type": "lithiumscope_model_bundle",
        "status": "candidate",
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "git_commit": model_1.get("git_commit"),
        "git_commits": {
            "model_1": model_1.get("git_commit"),
            "model_2": model_2.get("git_commit"),
        },
        "mixed_training_commits": (
            model_1.get("git_commit")
            != model_2.get("git_commit")
        ),
        "suggested_tag": suggested_tag,
        "model_1": model_1,
        "model_2": model_2,
        "note": (
            "Local model publication manifest. Each model preserves its own "
            "training provenance; a previously versioned active model may be "
            "reused without retraining. This manifest does not create a Git "
            "tag or GitHub Release."
        ),
    }

    destination = (
        RESULTS_DIR
        / "release_candidates"
        / f"{stamp}.json"
    )
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    destination.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )
    return destination


def _artifact_directory(
    model_group: str,
    run_id: str,
) -> Path:
    path = (
        MODELS_DIR
        / model_group
        / "trained"
        / run_id
    )
    required = (
        path / "model.joblib",
        path / "metadata.json",
    )
    if not all(item.is_file() for item in required):
        raise RuntimeError(
            "No se encontró el artefacto entrenado completo para "
            f"{model_group}: {path}"
        )
    return path


def build_release_bundle(
    manifest_path: Path,
) -> Path:
    try:
        manifest = json.loads(
            manifest_path.read_text(
                encoding="utf-8"
            )
        )
    except (
        OSError,
        json.JSONDecodeError,
    ) as exc:
        raise RuntimeError(
            f"Manifest inválido: {manifest_path}"
        ) from exc

    tag = str(
        manifest.get(
            "suggested_tag",
            "",
        )
    )
    if not tag:
        raise RuntimeError(
            "El manifest no contiene suggested_tag."
        )

    artifacts: dict[str, Path] = {}
    for model_group in (
        "model_1",
        "model_2",
    ):
        model_entry = manifest.get(
            model_group
        )
        if not isinstance(
            model_entry,
            dict,
        ):
            raise RuntimeError(
                f"Manifest incompleto para {model_group}."
            )
        run_id = str(
            model_entry.get(
                "run_id",
                "",
            )
        )
        if not run_id:
            raise RuntimeError(
                f"Manifest sin run_id para {model_group}."
            )
        artifacts[model_group] = (
            _artifact_directory(
                model_group,
                run_id,
            )
        )

    destination = (
        manifest_path.parent
        / f"{tag}-artifacts.zip"
    )
    temporary = destination.with_suffix(
        ".zip.tmp"
    )

    with zipfile.ZipFile(
        temporary,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        archive.write(
            manifest_path,
            arcname="release_manifest.json",
        )
        for model_group, directory in (
            artifacts.items()
        ):
            for path in sorted(
                item
                for item in directory.rglob("*")
                if item.is_file()
            ):
                relative = path.relative_to(
                    directory
                )
                archive.write(
                    path,
                    arcname=str(
                        Path(model_group)
                        / directory.name
                        / relative
                    ),
                )

    temporary.replace(
        destination
    )
    return destination


def prepare_release_candidate_bundle() -> tuple[Path, Path]:
    manifest_path = (
        build_release_candidate_manifest()
    )
    bundle_path = build_release_bundle(
        manifest_path
    )
    return manifest_path, bundle_path
