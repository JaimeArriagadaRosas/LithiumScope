from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import importlib.util

from lithiumscope.core.config import load_config
from lithiumscope.core.logger import get_logger
from lithiumscope.core.paths import DATA_DIR, PROJECT_ROOT
from lithiumscope.core.states import DatasetState
from lithiumscope.datasets.downloader import ensure_dataset
from lithiumscope.datasets.registry import get_dataset_spec
from lithiumscope.datasets.status import DatasetStatus
from lithiumscope.datasets.model_1_sources import (
    prepare_model_1_training_source,
)

logger = get_logger("datasets.provisioner")


@dataclass(frozen=True)
class TrainingDatasets:
    model_1: Path
    model_2_manifest: Path | None


def _missing_imagery_dependencies() -> list[str]:
    required = {
        "Rasterio": "rasterio",
        "pystac-client": "pystac_client",
    }
    return [
        name
        for name, import_name in required.items()
        if importlib.util.find_spec(import_name) is None
    ]


def _project_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else PROJECT_ROOT / path


def inspect_required_datasets() -> list[DatasetStatus]:
    """Inspect only local dataset state. This function never downloads data."""
    model_1_spec = get_dataset_spec("mamani09_public_mirror")
    model_1_path = (
        DATA_DIR
        / "raw"
        / model_1_spec.model
        / model_1_spec.destination_name
    )
    model_1_ready = (
        model_1_path.exists()
        and model_1_path.is_file()
        and model_1_path.stat().st_size > 0
    )

    model_2_cfg = load_config("model_2")
    manifest_path = _project_path(
        str(model_2_cfg["training"]["manifest_path"])
    )
    model_2_ready = (
        manifest_path.exists()
        and manifest_path.is_file()
        and manifest_path.stat().st_size > 0
    )

    return [
        DatasetStatus(
            key="model_1_geochemistry",
            ready=model_1_ready,
            path=str(model_1_path) if model_1_ready else None,
            detail=(
                "Dataset geoquímico local disponible."
                if model_1_ready
                else "No preparado todavía; se descargará al iniciar entrenamiento."
            ),
            state=DatasetState.READY if model_1_ready else DatasetState.MISSING,
        ),
        DatasetStatus(
            key="model_2_sentinel2",
            ready=model_2_ready,
            path=str(manifest_path) if model_2_ready else None,
            detail=(
                "Manifest espacial local disponible."
                if model_2_ready
                else "No preparado todavía; se construirá al iniciar entrenamiento."
            ),
            state=DatasetState.READY if model_2_ready else DatasetState.MISSING,
        ),
    ]


def provision_required_datasets(
    prepare_model_2: bool = True,
) -> list[DatasetStatus]:
    statuses: list[DatasetStatus] = []

    try:
        base_model_1 = ensure_dataset("mamani09_public_mirror")
        model_1 = prepare_model_1_training_source(
            base_model_1
        )
        statuses.append(
            DatasetStatus(
                key="model_1_geochemistry",
                ready=True,
                path=str(model_1),
                detail=(
                    "Dataset geoquímico disponible"
                    + (
                        " con fuentes adicionales armonizadas."
                        if model_1 != base_model_1
                        else " (bootstrap base)."
                    )
                ),
                state=DatasetState.READY,
            )
        )
    except Exception as exc:
        logger.exception("Could not provision Model 1 dataset")
        statuses.append(
            DatasetStatus(
                key="model_1_geochemistry",
                ready=False,
                path=None,
                detail=str(exc),
                state=DatasetState.FAILED,
            )
        )
        return statuses

    if not prepare_model_2:
        return statuses

    missing_imagery = _missing_imagery_dependencies()
    if missing_imagery:
        statuses.append(
            DatasetStatus(
                key="model_2_sentinel2",
                ready=False,
                path=None,
                detail=(
                    "Dependencias de imágenes faltantes: "
                    + ", ".join(missing_imagery)
                ),
                state=DatasetState.MISSING,
            )
        )
        return statuses

    try:
        from lithiumscope.model_2.data.builder import (
            ensure_model_2_dataset,
        )

        result = ensure_model_2_dataset(model_1)
        statuses.append(
            DatasetStatus(
                key="model_2_sentinel2",
                ready=True,
                path=str(result.manifest_path),
                detail=(
                    f"Pares listos: {result.ready_samples}; "
                    f"omitidos: {result.failed_samples}; "
                    f"cache: {result.resumed_samples}."
                ),
                state=DatasetState.READY,
            )
        )
    except Exception as exc:
        logger.error(
            "Model 2 Sentinel provisioning failed: %s",
            exc,
        )
        statuses.append(
            DatasetStatus(
                key="model_2_sentinel2",
                ready=False,
                path=None,
                detail=str(exc),
                state=DatasetState.FAILED,
            )
        )

    return statuses


def _status_map(
    prepare_model_2: bool = True,
) -> dict[str, DatasetStatus]:
    return {
        status.key: status
        for status in provision_required_datasets(
            prepare_model_2=prepare_model_2
        )
    }


def require_model_1_dataset() -> Path:
    status = _status_map(
        prepare_model_2=False
    ).get("model_1_geochemistry")
    if status is None or not status.ready or not status.path:
        detail = status.detail if status else "sin estado"
        raise RuntimeError(
            "Dataset automático de Modelo 1 no disponible: "
            f"{detail}."
        )
    return Path(status.path)


def require_model_2_dataset() -> Path:
    statuses = _status_map(prepare_model_2=True)
    status = statuses.get("model_2_sentinel2")
    if status is None or not status.ready or not status.path:
        detail = status.detail if status else "sin estado"
        raise RuntimeError(
            "Dataset automático de Modelo 2 no disponible: "
            f"{detail}."
        )
    return Path(status.path)


def require_training_datasets() -> TrainingDatasets:
    model_1 = require_model_1_dataset()
    model_2 = require_model_2_dataset()
    return TrainingDatasets(
        model_1=model_1,
        model_2_manifest=model_2,
    )
