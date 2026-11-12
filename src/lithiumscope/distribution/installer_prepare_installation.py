"""Separated _prepare_installation responsibility; preserve prior public API."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lithiumscope.distribution.installer import InstalledModel, Path

def _prepare_installation(
    model_group: str,
    artifact_dir: Path,
    metadata: dict,
) -> tuple[InstalledModel, Path | None, Path]:
    from lithiumscope.distribution.installer import InstalledModel, MODELS_DIR, Path, file_sha256, shutil
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
