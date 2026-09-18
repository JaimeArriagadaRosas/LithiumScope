from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import importlib.util

from lithiumscope.core.logger import get_logger
from lithiumscope.datasets.downloader import ensure_dataset
from lithiumscope.datasets.status import DatasetStatus

logger = get_logger("datasets.provisioner")


@dataclass(frozen=True)
class TrainingDatasets:
    model_1: Path
    model_2_manifest: Path | None


def provision_required_datasets(prepare_model_2: bool = True) -> list[DatasetStatus]:
    statuses: list[DatasetStatus] = []

    try:
        model_1 = ensure_dataset("mamani09_public_mirror")
        statuses.append(
            DatasetStatus(
                key="model_1_geochemistry",
                ready=True,
                path=str(model_1),
                detail="Dataset geoquímico bootstrap disponible.",
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
            )
        )
        return statuses

    if not prepare_model_2:
        return statuses

    if importlib.util.find_spec("rasterio") is None:
        statuses.append(
            DatasetStatus(
                key="model_2_sentinel2",
                ready=False,
                path=None,
                detail=(
                    "Rasterio no está instalado; no se puede preparar automáticamente "
                    "el dataset Sentinel-2."
                ),
            )
        )
        return statuses

    try:
        from lithiumscope.model_2.data.builder import ensure_model_2_dataset

        result = ensure_model_2_dataset(model_1)
        statuses.append(
            DatasetStatus(
                key="model_2_sentinel2",
                ready=True,
                path=str(result.manifest_path),
                detail=(
                    f"Pares muestra-imagen listos: {result.ready_samples}; "
                    f"omitidos: {result.failed_samples}."
                ),
            )
        )
    except Exception as exc:
        logger.exception("Could not provision Model 2 Sentinel-2 dataset")
        statuses.append(
            DatasetStatus(
                key="model_2_sentinel2",
                ready=False,
                path=None,
                detail=str(exc),
            )
        )

    return statuses


def _status_map(prepare_model_2: bool = True) -> dict[str, DatasetStatus]:
    return {
        status.key: status
        for status in provision_required_datasets(prepare_model_2=prepare_model_2)
    }


def require_model_1_dataset() -> Path:
    status = _status_map(prepare_model_2=False).get("model_1_geochemistry")
    if status is None or not status.ready or not status.path:
        detail = status.detail if status else "sin estado"
        raise RuntimeError(
            "Dataset automático de Modelo 1 no disponible: "
            f"{detail}. Revise logs/preboot y la conexión a Internet."
        )
    return Path(status.path)


def require_model_2_dataset() -> Path:
    statuses = _status_map(prepare_model_2=True)
    status = statuses.get("model_2_sentinel2")
    if status is None or not status.ready or not status.path:
        detail = status.detail if status else "sin estado"
        raise RuntimeError(
            "Dataset automático de Modelo 2 no disponible: "
            f"{detail}. Revise logs/preboot y la conexión a Internet."
        )
    return Path(status.path)


def require_training_datasets() -> TrainingDatasets:
    model_1 = require_model_1_dataset()
    model_2 = require_model_2_dataset()
    return TrainingDatasets(model_1=model_1, model_2_manifest=model_2)
