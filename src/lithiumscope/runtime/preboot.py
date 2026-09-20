from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import importlib.util
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
from lithiumscope.core.visualization import configure_headless_matplotlib

logger = get_logger("runtime.preboot")

CORE_DEPENDENCIES = {
    "joblib": "joblib",
    "matplotlib": "matplotlib",
    "numpy": "numpy",
    "openpyxl": "openpyxl",
    "pandas": "pandas",
    "PyYAML": "yaml",
    "requests": "requests",
    "ReportLab": "reportlab",
    "scikit-learn": "sklearn",
}

ML_DEPENDENCIES = {
    "CatBoost": "catboost",
    "Optuna": "optuna",
    "PyTorch": "torch",
    "PyTorch TabNet": "pytorch_tabnet",
    "XGBoost": "xgboost",
}

IMAGERY_DEPENDENCIES = {
    "Pillow": "PIL",
    "Rasterio": "rasterio",
    "pystac-client": "pystac_client",
}

DEV_DEPENDENCIES = {
    "pytest": "pytest",
    "ruff": "ruff",
}

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


@dataclass(frozen=True)
class DependencyStatus:
    name: str
    import_name: str
    available: bool


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
    virtualenv_active: bool = False
    visualization_backend: str = "unknown"
    report_path: str = ""

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
    return [
        DependencyStatus(
            name,
            import_name,
            importlib.util.find_spec(import_name) is not None,
        )
        for name, import_name in mapping.items()
    ]


def _missing(
    items: list[DependencyStatus],
) -> list[str]:
    return [
        item.name
        for item in items
        if not item.available
    ]


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


def _save_report(
    report: PrebootReport,
) -> Path:
    directory = LOGS_DIR / "preboot"
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )
    stamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")
    path = directory / f"preboot_{stamp}.json"
    report.report_path = str(path)
    payload = asdict(report)
    payload["core_ready"] = report.core_ready
    payload["ml_ready"] = report.ml_ready
    payload["imagery_ready"] = report.imagery_ready
    payload["datasets_ready"] = report.datasets_ready
    payload["model_1_ready"] = report.model_1_ready
    payload["model_2_ready"] = report.model_2_ready
    payload["training_ready"] = report.training_ready
    payload["session_log"] = str(
        current_session_log_path()
    )
    payload["error_log"] = str(
        current_error_log_path()
    )
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path


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
        state = "OK" if item.ready else "ERROR"
        print(
            f"    - {item.key}: {state} — {item.detail}"
        )


def run_preboot(
    verbose: bool = True,
    *,
    provision_datasets: bool = False,
) -> PrebootReport:
    ensure_runtime_directories()
    configure_logging()

    removed = cleanup_gitkeep_placeholders()
    python_ok = sys.version_info >= MIN_PYTHON
    configs_ok = _configs_available()
    writable_ok = _runtime_writable()
    core = _dependency_status(CORE_DEPENDENCIES)
    ml = _dependency_status(ML_DEPENDENCIES)
    imagery = _dependency_status(
        IMAGERY_DEPENDENCIES
    )
    dev = _dependency_status(
        DEV_DEPENDENCIES
    )
    device = detect_device(prefer_gpu=True)
    virtualenv_active = _virtualenv_active()
    visualization_backend = configure_headless_matplotlib()

    if verbose:
        print(
            "\n=== PREBOOT LITHIUMSCOPE: "
            "CHEQUEOS RÁPIDOS ==="
        )
        print(
            f"  Python             "
            f"{platform.python_version()} "
            f"[{'OK' if python_ok else 'INCOMPATIBLE'}]"
        )
        print(
            f"  Entorno virtual    "
            f"[{'ACTIVO' if virtualenv_active else 'NO ACTIVO'}]"
        )
        print(
            f"  Configuración      "
            f"[{'OK' if configs_ok else 'ERROR'}]"
        )
        print(
            f"  Directorios        "
            f"[{'OK' if writable_ok else 'ERROR'}]"
        )
        _print_group("Core", core)
        _print_group("ML completo", ml)
        _print_group("Imágenes", imagery)
        _print_group("Desarrollo", dev)
        print(
            f"  Acelerador         "
            f"{device.accelerator.upper()} — {device.name}"
        )
        print(
            f"  Visualización      [OK] {visualization_backend}"
        )
        if not virtualenv_active:
            print(
                "  Aviso              "
                "Se está usando el Python global; "
                "se recomienda .venv."
            )
        print(
            "\n  Revisando estado local de datasets..."
        )
        if not provision_datasets:
            print(
                "  Nota               Los datasets se preparan al elegir Entrenar."
            )

    can_prepare_data = (
        configs_ok
        and all(
            item.available
            for item in core
        )
    )
    datasets = (
        (
            provision_required_datasets(
                prepare_model_2=all(
                    item.available
                    for item in imagery
                )
            )
            if provision_datasets
            else inspect_required_datasets()
        )
        if can_prepare_data
        else []
    )

    report = PrebootReport(
        python_ok=python_ok,
        python_version=platform.python_version(),
        platform=(
            f"{platform.system()} "
            f"{platform.release()}"
        ),
        core=core,
        ml=ml,
        imagery=imagery,
        dev=dev,
        datasets=datasets,
        configs_ok=configs_ok,
        writable_ok=writable_ok,
        placeholders_removed=removed,
        accelerator=device.accelerator,
        accelerator_name=device.name,
        virtualenv_active=virtualenv_active,
        visualization_backend=visualization_backend,
    )
    _save_report(report)

    logger.info(
        "Preboot complete core=%s ml=%s imagery=%s "
        "model1=%s model2=%s venv=%s matplotlib=%s",
        report.core_ready,
        report.ml_ready,
        report.imagery_ready,
        report.model_1_ready,
        report.model_2_ready,
        report.virtualenv_active,
        report.visualization_backend,
    )

    if verbose:
        print(
            "\n=== PREBOOT LITHIUMSCOPE: RESULTADO ==="
        )
        print(
            f"  Plataforma         {report.platform}"
        )
        print(
            f"  .gitkeep limpiados "
            f"{report.placeholders_removed}"
        )
        _print_datasets(report.datasets)
        print(
            f"  Entrenamiento M1   "
            f"[{'LISTO' if report.model_1_ready else 'DATOS PENDIENTES'}]"
        )
        print(
            f"  Entrenamiento M2   "
            f"[{'LISTO' if report.model_2_ready else 'DATOS PENDIENTES'}]"
        )
        print(
            f"  Reporte            {report.report_path}"
        )
        print(
            f"  Log de sesión      "
            f"{current_session_log_path()}"
        )
        if not report.ml_ready or not report.imagery_ready:
            print(
                "\n  Para dejar el entorno completo:"
            )
            print(
                '    pip install -e ".[ml,imagery,dev]"'
            )
        print("=" * 43)

    return report


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
