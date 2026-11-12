from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
import tempfile

from lithiumscope.core.device import detect_device
from lithiumscope.core.logger import (
    configure_logging,
    current_error_log_path,
    current_session_log_path,
    get_logger,
)
from lithiumscope.core.paths import (
    CONFIG_DIR,
    DATA_DIR,
    LOGS_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    ensure_runtime_directories,
)
from lithiumscope.datasets.provisioner import inspect_required_datasets, provision_required_datasets
from lithiumscope.datasets.status import DatasetStatus
from lithiumscope.core.states import DatasetState
from lithiumscope.core.visualization import configure_headless_matplotlib
from lithiumscope.runtime.console_status import LoadingBar
from lithiumscope.runtime.dependencies import (
    CORE_DEPENDENCIES,
    DEV_DEPENDENCIES,
    IMAGERY_DEPENDENCIES,
    ML_DEPENDENCIES,
    DependencyStatus,
    dependency_status,
    ensure_runtime_dependencies,
    missing_names,
)

logger = get_logger("runtime.preboot")

REQUIRED_CONFIGS = (
    "app.yaml",
    "distribution.yaml",
    "logging.yaml",
    "model_1.yaml",
    "model_2.yaml",
    "prediction.yaml",
)
REQUIRED_DATASETS = {
    "model_1_geochemistry",
    "model_2_sentinel2",
}
MIN_PYTHON = (3, 11)


@dataclass
class PrebootReport:
    python_ok: bool
    python_version: str
    platform: str
    core: list[DependencyStatus]
    ml: list[DependencyStatus]
    imagery: list[DependencyStatus]
    dev: list[DependencyStatus]
    datasets: list[DatasetStatus]
    configs_ok: bool
    writable_ok: bool
    placeholders_removed: int
    accelerator: str
    accelerator_name: str
    accelerator_diagnostic: str = ""
    nvidia_driver_available: bool = False
    nvidia_device_names: tuple[str, ...] = ()
    xgboost_cuda_available: bool = False
    virtualenv_active: bool = False
    visualization_backend: str = "unknown"
    report_path: str = ""
    dependency_repair: dict = field(default_factory=dict)

    @property
    def core_ready(self) -> bool:
        return (
            self.python_ok
            and self.configs_ok
            and self.writable_ok
            and all(item.available for item in self.core)
        )

    @property
    def ml_ready(self) -> bool:
        return all(item.available for item in self.ml)

    @property
    def imagery_ready(self) -> bool:
        return all(item.available for item in self.imagery)

    def dataset_ready(self, key: str) -> bool:
        return any(
            item.key == key and item.ready
            for item in self.datasets
        )

    @property
    def model_1_ready(self) -> bool:
        return (
            self.core_ready
            and self.ml_ready
            and self.dataset_ready("model_1_geochemistry")
        )

    @property
    def model_2_ready(self) -> bool:
        return (
            self.core_ready
            and self.ml_ready
            and self.imagery_ready
            and self.dataset_ready("model_2_sentinel2")
        )

    @property
    def datasets_ready(self) -> bool:
        return all(
            self.dataset_ready(key)
            for key in REQUIRED_DATASETS
        )

    @property
    def training_ready(self) -> bool:
        return self.model_1_ready and self.model_2_ready


def _dependency_status(
    mapping: dict[str, str],
) -> list[DependencyStatus]:
    return dependency_status(mapping)


def _missing(
    items: list[DependencyStatus],
) -> list[str]:
    return missing_names(items)


def _virtualenv_active() -> bool:
    return sys.prefix != getattr(
        sys,
        "base_prefix",
        sys.prefix,
    )


def cleanup_gitkeep_placeholders() -> int:
    """Remove cloned .gitkeep files from the local runtime workspace."""
    removed = 0
    for root in (
        DATA_DIR,
        MODELS_DIR,
        RESULTS_DIR,
        LOGS_DIR,
    ):
        if not root.exists():
            continue
        for path in root.rglob(".gitkeep"):
            try:
                path.unlink()
                removed += 1
                logger.info(
                    "Removed local runtime placeholder: %s",
                    path,
                )
            except OSError:
                logger.warning(
                    "Could not remove local placeholder: %s",
                    path,
                    exc_info=True,
                )
    return removed


def _configs_available() -> bool:
    missing = [
        name
        for name in REQUIRED_CONFIGS
        if not (CONFIG_DIR / name).is_file()
    ]
    if missing:
        logger.error(
            "Missing configuration files: %s",
            ", ".join(missing),
        )
        return False
    return True


def _runtime_writable() -> bool:
    targets = (
        DATA_DIR,
        MODELS_DIR,
        RESULTS_DIR,
        LOGS_DIR,
    )
    try:
        for directory in targets:
            directory.mkdir(
                parents=True,
                exist_ok=True,
            )
            with tempfile.NamedTemporaryFile(
                dir=directory,
                prefix=".write_test_",
                delete=True,
            ):
                pass
        return True
    except OSError:
        logger.exception(
            "A runtime directory is not writable"
        )
        return False


from lithiumscope.runtime.preboot_save_report import _save_report


def _print_group(
    title: str,
    items: list[DependencyStatus],
) -> None:
    missing = _missing(items)
    mark = "OK" if not missing else "FALTAN"
    print(f"  {title:<18} [{mark}]")
    if missing:
        print("    - " + ", ".join(missing))


def _print_datasets(
    items: list[DatasetStatus],
) -> None:
    by_key = {
        item.key: item
        for item in items
    }
    print("  Datasets")
    for key in sorted(REQUIRED_DATASETS):
        item = by_key.get(key)
        if item is None:
            print(f"    - {key}: NO PREPARADO")
            continue
        state = ("OK" if item.ready else "PENDIENTE" if item.effective_state == DatasetState.MISSING else "ERROR")
        print(
            f"    - {item.key}: {state} — {item.detail}"
        )


from lithiumscope.runtime.preboot_run_preboot import run_preboot


def missing_training_dependencies() -> list[str]:
    return _missing(
        _dependency_status(ML_DEPENDENCIES)
    )


def require_full_training_environment() -> None:
    missing = missing_training_dependencies()
    if missing:
        raise RuntimeError(
            "El entorno no está listo para ejecutar "
            "la competencia completa. "
            f"Faltan: {', '.join(missing)}. "
            'Ejecute: pip install -e ".[ml,imagery,dev]" '
            "y reinicie LithiumScope."
        )
