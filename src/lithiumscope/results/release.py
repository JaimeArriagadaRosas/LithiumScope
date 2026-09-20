from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import zipfile

from lithiumscope.core.paths import MODELS_DIR, RESULTS_DIR
from lithiumscope.results.catalog import latest_release_candidate


def build_release_candidate_manifest() -> Path:
    model_1 = latest_release_candidate("model_1")
    model_2 = latest_release_candidate("model_2")

    if model_1 is None or model_2 is None:
        missing = []
        if model_1 is None:
            missing.append("model_1")
        if model_2 is None:
            missing.append("model_2")
        raise RuntimeError(
            "No existe una ejecución completa apta para versionar en: "
            + ", ".join(missing)
        )

    if model_1["git_commit"] != model_2["git_commit"]:
        raise RuntimeError(
            "Los candidatos de Modelo 1 y Modelo 2 fueron generados con commits "
            "distintos. Ejecute ambos modelos sobre el mismo código antes de publicar."
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
        "git_commit": model_1["git_commit"],
        "suggested_tag": suggested_tag,
        "model_1": model_1,
        "model_2": model_2,
        "note": (
            "Local model publication manifest. "
            "It does not create a Git tag or GitHub Release."
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
