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
from lithiumscope.core.logger import configure_logging, get_logger
from lithiumscope.core.paths import (
    CONFIG_DIR,
    DATA_DIR,
    LOGS_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    ensure_runtime_directories,
)
from lithiumscope.datasets.provisioner import provision_required_datasets
from lithiumscope.datasets.status import DatasetStatus

logger = get_logger("runtime.preboot")

CORE_DEPENDENCIES = {
    "joblib": "joblib",
    "matplotlib": "matplotlib",
    "numpy": "numpy",
    "openpyxl": "openpyxl",
    "pandas": "pandas",
    "PyYAML": "yaml",
    "requests": "requests",
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
}

DEV_DEPENDENCIES = {
    "pytest": "pytest",
    "ruff": "ruff",
}

REQUIRED_CONFIGS = ("app.yaml", "logging.yaml", "model_1.yaml", "model_2.yaml")
REQUIRED_DATASETS = {"model_1_geochemistry", "model_2_sentinel2"}
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

    @property
    def datasets_ready(self) -> bool:
        by_key = {item.key: item for item in self.datasets}
        return all(
            key in by_key and by_key[key].ready
            for key in REQUIRED_DATASETS
        )

    @property
    def training_ready(self) -> bool:
        return self.core_ready and self.ml_ready and self.imagery_ready and self.datasets_ready


def _dependency_status(mapping: dict[str, str]) -> list[DependencyStatus]:
    return [
        DependencyStatus(name, import_name, importlib.util.find_spec(import_name) is not None)
        for name, import_name in mapping.items()
    ]


def _missing(items: list[DependencyStatus]) -> list[str]:
    return [item.name for item in items if not item.available]


def cleanup_gitkeep_placeholders() -> int:
    """Remove cloned .gitkeep files from the local runtime workspace.

    The repository intentionally tracks these files so GitHub and a fresh clone
    display the complete intended directory architecture. Once LithiumScope is
    executed, preboot removes the local placeholder files because the runtime
    directories are already materialized and will contain real artifacts.
    """
    removed = 0
    for root in (DATA_DIR, MODELS_DIR, RESULTS_DIR, LOGS_DIR):
        if not root.exists():
            continue
        for path in root.rglob(".gitkeep"):
            try:
                path.unlink()
                removed += 1
                logger.info("Removed local runtime placeholder: %s", path)
            except OSError:
                logger.warning("Could not remove local placeholder: %s", path, exc_info=True)
    return removed


def _configs_available() -> bool:
    missing = [name for name in REQUIRED_CONFIGS if not (CONFIG_DIR / name).is_file()]
    if missing:
        logger.error("Missing configuration files: %s", ", ".join(missing))
        return False
    return True


def _runtime_writable() -> bool:
    targets = (DATA_DIR, MODELS_DIR, RESULTS_DIR, LOGS_DIR)
    try:
        for directory in targets:
            directory.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=directory, prefix=".write_test_", delete=True):
                pass
        return True
    except OSError:
        logger.exception("A runtime directory is not writable")
        return False


def _save_report(report: PrebootReport) -> Path:
    directory = LOGS_DIR / "preboot"
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = directory / f"preboot_{stamp}.json"
    payload = asdict(report)
    payload["core_ready"] = report.core_ready
    payload["ml_ready"] = report.ml_ready
    payload["imagery_ready"] = report.imagery_ready
    payload["datasets_ready"] = report.datasets_ready
    payload["training_ready"] = report.training_ready
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _print_group(title: str, items: list[DependencyStatus]) -> None:
    missing = _missing(items)
    mark = "OK" if not missing else "FALTAN"
    print(f"  {title:<18} [{mark}]")
    if missing:
        print("    - " + ", ".join(missing))


def _print_datasets(items: list[DatasetStatus]) -> None:
    by_key = {item.key: item for item in items}
    complete = all(key in by_key and by_key[key].ready for key in REQUIRED_DATASETS)
    print(f"  Datasets           [{'OK' if complete else 'INCOMPLETOS'}]")
    for key in sorted(REQUIRED_DATASETS):
        item = by_key.get(key)
        if item is None:
            print(f"    - {key}: NO PREPARADO")
            continue
        state = "OK" if item.ready else "ERROR"
        print(f"    - {item.key}: {state} — {item.detail}")


def run_preboot(verbose: bool = True) -> PrebootReport:
    ensure_runtime_directories()
    configure_logging()

    removed = cleanup_gitkeep_placeholders()
    python_ok = sys.version_info >= MIN_PYTHON
    configs_ok = _configs_available()
    writable_ok = _runtime_writable()
    core = _dependency_status(CORE_DEPENDENCIES)
    ml = _dependency_status(ML_DEPENDENCIES)
    imagery = _dependency_status(IMAGERY_DEPENDENCIES)
    dev = _dependency_status(DEV_DEPENDENCIES)
    device = detect_device(prefer_gpu=True)

    can_prepare_data = all(item.available for item in core)
    datasets = (
        provision_required_datasets(prepare_model_2=all(item.available for item in imagery))
        if can_prepare_data
        else []
    )

    report = PrebootReport(
        python_ok=python_ok,
        python_version=platform.python_version(),
        platform=f"{platform.system()} {platform.release()}",
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
    )
    report.report_path = str(_save_report(report))

    logger.info(
        "Preboot complete core=%s ml=%s imagery=%s datasets=%s training=%s",
        report.core_ready,
        report.ml_ready,
        report.imagery_ready,
        report.datasets_ready,
        report.training_ready,
    )

    if verbose:
        print("\n=== PREBOOT LITHIUMSCOPE ===")
        print(f"  Python             {report.python_version} [{'OK' if report.python_ok else 'INCOMPATIBLE'}]")
        print(f"  Plataforma         {report.platform}")
        print(f"  Acelerador         {report.accelerator.upper()} — {report.accelerator_name}")
        print(f"  Configuración      [{'OK' if report.configs_ok else 'ERROR'}]")
        print(f"  Directorios        [{'OK' if report.writable_ok else 'ERROR'}]")
        print(f"  .gitkeep limpiados {report.placeholders_removed}")
        _print_group("Core", report.core)
        _print_group("ML completo", report.ml)
        _print_group("Imágenes", report.imagery)
        _print_group("Desarrollo", report.dev)
        _print_datasets(report.datasets)
        print(f"  Entrenamiento      [{'LISTO' if report.training_ready else 'NO LISTO'}]")
        print(f"  Reporte            {report.report_path}")
        if not report.ml_ready or not report.imagery_ready:
            print("\n  Para dejar el entorno completo:")
            print('    pip install -e ".[ml,imagery,dev]"')
        print("=" * 29)

    return report


def missing_training_dependencies() -> list[str]:
    return _missing(_dependency_status(ML_DEPENDENCIES))


def require_full_training_environment() -> None:
    missing = missing_training_dependencies()
    if missing:
        raise RuntimeError(
            "El entorno no está listo para ejecutar la competencia completa. "
            f"Faltan: {', '.join(missing)}. "
            'Ejecute: pip install -e ".[ml,imagery,dev]" y reinicie LithiumScope.'
        )
