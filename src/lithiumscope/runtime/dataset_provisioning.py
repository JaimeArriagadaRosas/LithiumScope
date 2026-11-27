from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import importlib.util

from lithiumscope.core.config import load_config
from lithiumscope.core.logging.session import get_logger
from lithiumscope.core.paths import DATA_DIR, PROJECT_ROOT
from lithiumscope.core.states import DatasetState
from lithiumscope.tools.acquisition.tabular import ensure_dataset
from lithiumscope.datasets.management.status import DatasetStatus
from lithiumscope.datasets.management.sources.model_1 import (
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
    """Resolve the configured cohort without downloading or reconstructing it."""
    from lithiumscope.datasets.management.sources.local import local_training_source
    statuses = []
    for model, key in (("model_1", "model_1_geochemistry"), ("model_2", "model_2_sentinel2")):
        try:
            path = local_training_source(model, config=load_config(model), root=PROJECT_ROOT, data_dir=DATA_DIR)
            statuses.append(DatasetStatus(key, True, str(path), "Fuente configurada verificada.",
                                          DatasetState.READY))
        except FileNotFoundError as exc:
            statuses.append(DatasetStatus(key, False, None, str(exc), DatasetState.MISSING))
        except (ValueError, RuntimeError) as exc:
            statuses.append(DatasetStatus(key, False, None, str(exc), DatasetState.CORRUPTED))
        except (OSError, KeyError) as exc:
            statuses.append(DatasetStatus(key, False, None, str(exc), DatasetState.FAILED))
    return statuses

def provision_required_datasets(
    prepare_model_2: bool = True,
    *,
    include_georoc: bool | None = None,
) -> list[DatasetStatus]:
    statuses: list[DatasetStatus] = []

    try:
        cfg = load_config("model_1")
        georoc = cfg.get("data_sources",{}).get("georoc",{})
        use_georoc = georoc.get("enabled",False) if include_georoc is None else include_georoc
        if use_georoc and georoc.get("frozen_combined_sha256"):
            # The verified combined source does not need a duplicate bootstrap download.
            base_model_1 = _project_path(str(georoc["combined_path"]))
        else:
            base_model_1 = ensure_dataset(cfg.get("data_sources",{}).get("base",{}).get("key","mamani09_public_mirror"))
        model_1 = prepare_model_1_training_source(
            base_model_1,
            include_georoc=include_georoc,
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
    *,
    include_georoc: bool | None = None,
) -> dict[str, DatasetStatus]:
    return {
        status.key: status
        for status in provision_required_datasets(
            prepare_model_2=prepare_model_2,
            include_georoc=include_georoc,
        )
    }


def require_model_1_dataset(
    *,
    include_georoc: bool | None = None,
) -> Path:
    status = _status_map(
        prepare_model_2=False,
        include_georoc=include_georoc,
    ).get("model_1_geochemistry")
    if status is None or not status.ready or not status.path:
        detail = status.detail if status else "sin estado"
        raise RuntimeError(
            "Dataset automático de Modelo 1 no disponible: "
            f"{detail}."
        )
    return Path(status.path)


def require_frozen_model_2_manifest() -> Path:
    from lithiumscope.datasets.management.sources.model_2 import require_frozen_manifest
    return require_frozen_manifest(load_config("model_2")["training"], PROJECT_ROOT)


def require_model_2_expanded_manifest() -> Path:
    """Prepare isolated M2-only extension; preserve M1 and legacy M2."""
    config = load_config("model_2")
    if not config["training"].get("expansion", {}).get("enabled", False):
        return require_frozen_model_2_manifest()
    from lithiumscope.model_2.data.expansion_vnext import (
        ensure_expanded_m2_manifest,
    )
    path = ensure_expanded_m2_manifest()
    if path is None:
        raise RuntimeError("M2 expansion not ready; nothing will be trained")
    from lithiumscope.datasets.management.sources.model_2 import require_expanded_manifest
    return require_expanded_manifest(config["training"], PROJECT_ROOT)


def require_model_2_dataset(
    *,
    include_georoc: bool | None = None,
) -> Path:
    training = load_config("model_2")["training"]
    if training.get("expansion", {}).get("enabled"):
        return require_model_2_expanded_manifest()
    if training.get("frozen_manifest_sha256"):
        return require_frozen_model_2_manifest()
    statuses = _status_map(
        prepare_model_2=True,
        include_georoc=include_georoc,
    )
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
