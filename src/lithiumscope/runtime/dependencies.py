from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

from lithiumscope.runtime.console_status import LoadingBar


PROJECT_ROOT = Path(__file__).resolve().parents[3]

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


@dataclass(frozen=True)
class DependencyStatus:
    name: str
    import_name: str
    available: bool


@dataclass(frozen=True)
class DependencyRepairResult:
    attempted: bool
    success: bool
    missing_before: tuple[str, ...]
    missing_after: tuple[str, ...]
    extras: tuple[str, ...] = ()
    command: tuple[str, ...] = ()
    returncode: int | None = None
    skipped_reason: str | None = None
    installer_output_tail: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return asdict(self)


_LAST_REPAIR_RESULT: DependencyRepairResult | None = None


def dependency_status(
    mapping: dict[str, str],
) -> list[DependencyStatus]:
    return [
        DependencyStatus(
            name=name,
            import_name=import_name,
            available=importlib.util.find_spec(import_name) is not None,
        )
        for name, import_name in mapping.items()
    ]


def missing_names(
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


def _runtime_gaps() -> tuple[list[str], list[str], list[str]]:
    return (
        missing_names(dependency_status(CORE_DEPENDENCIES)),
        missing_names(dependency_status(ML_DEPENDENCIES)),
        missing_names(dependency_status(IMAGERY_DEPENDENCIES)),
    )


def _install_target(
    *,
    ml_missing: bool,
    imagery_missing: bool,
) -> tuple[str, tuple[str, ...]]:
    extras: list[str] = []
    if ml_missing:
        extras.append("ml")
    if imagery_missing:
        extras.append("imagery")
    suffix = f"[{','.join(extras)}]" if extras else ""
    return f".{suffix}", tuple(extras)


def last_dependency_repair() -> DependencyRepairResult | None:
    return _LAST_REPAIR_RESULT


def ensure_runtime_dependencies(
    *,
    verbose: bool = True,
    auto_install: bool = True,
) -> DependencyRepairResult:
    """Repair missing runtime dependencies before the full application imports.

    Base dependencies are installed with the editable project. Optional ML and
    imagery extras are requested only when their corresponding imports are
    missing. Development-only tools are deliberately excluded.
    """
    global _LAST_REPAIR_RESULT

    core_missing, ml_missing, imagery_missing = _runtime_gaps()
    missing_before = tuple(core_missing + ml_missing + imagery_missing)

    if not missing_before:
        if _LAST_REPAIR_RESULT is not None:
            return _LAST_REPAIR_RESULT
        result = DependencyRepairResult(
            attempted=False,
            success=True,
            missing_before=(),
            missing_after=(),
        )
        _LAST_REPAIR_RESULT = result
        return result

    if not auto_install or os.environ.get(
        "LITHIUMSCOPE_DISABLE_AUTO_INSTALL",
        "",
    ).strip().lower() in {"1", "true", "yes", "on"}:
        result = DependencyRepairResult(
            attempted=False,
            success=False,
            missing_before=missing_before,
            missing_after=missing_before,
            skipped_reason="auto_install_disabled",
        )
        _LAST_REPAIR_RESULT = result
        return result

    if not _virtualenv_active():
        result = DependencyRepairResult(
            attempted=False,
            success=False,
            missing_before=missing_before,
            missing_after=missing_before,
            skipped_reason="virtualenv_not_active",
        )
        _LAST_REPAIR_RESULT = result
        if verbose:
            print(
                "\nDependencias faltantes detectadas, pero no se modificará "
                "el Python global automáticamente."
            )
            print(
                'Active un .venv y ejecute de nuevo: python main.py'
            )
        return result

    target, extras = _install_target(
        ml_missing=bool(ml_missing),
        imagery_missing=bool(imagery_missing),
    )
    command = (
        sys.executable,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "-e",
        target,
    )

    loading = None
    if verbose:
        print("\n=== LITHIUMSCOPE · REPARACIÓN DE ENTORNO ===")
        print(
            "  Faltan             "
            + ", ".join(missing_before)
        )
        loading = LoadingBar(
            "Instalando dependencias declaradas"
        ).start()

    output = ""
    try:
        completed = subprocess.run(
            list(command),
            cwd=PROJECT_ROOT,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        returncode = int(completed.returncode)
        output = str(
            getattr(
                completed,
                "stdout",
                "",
            )
            or ""
        )
    except OSError as exc:
        returncode = -1
        output = str(exc)

    importlib.invalidate_caches()
    core_after, ml_after, imagery_after = _runtime_gaps()
    missing_after = tuple(
        core_after + ml_after + imagery_after
    )
    success = returncode == 0 and not missing_after

    output_lines = tuple(
        line.strip()
        for line in output.splitlines()
        if line.strip()
    )
    output_tail = output_lines[-12:]
    result = DependencyRepairResult(
        attempted=True,
        success=success,
        missing_before=missing_before,
        missing_after=missing_after,
        extras=extras,
        command=command,
        returncode=returncode,
        installer_output_tail=output_tail,
    )
    _LAST_REPAIR_RESULT = result

    if verbose:
        if loading is not None:
            if success:
                loading.succeed(
                    "Dependencias instaladas y verificadas"
                )
            else:
                loading.fail(
                    "No fue posible completar la instalación"
                )
        if missing_after:
            print(
                "  Aún faltan         "
                + ", ".join(missing_after)
            )
        if not success and output_tail:
            print(
                "  Detalle instalador:"
            )
            for line in output_tail:
                print(
                    "    " + line
                )
        print("=" * 44)

    return result
