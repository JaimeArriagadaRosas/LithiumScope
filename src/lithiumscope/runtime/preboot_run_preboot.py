"""Separated run_preboot responsibility; preserve prior public API."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lithiumscope.runtime.preboot import PrebootReport

def run_preboot(
    verbose: bool = True,
    *,
    provision_datasets: bool = False,
) -> PrebootReport:
    from lithiumscope.runtime.preboot import CORE_DEPENDENCIES, DEV_DEPENDENCIES, IMAGERY_DEPENDENCIES, LoadingBar, MIN_PYTHON, ML_DEPENDENCIES, PrebootReport, _configs_available, _dependency_status, _print_datasets, _print_group, _runtime_writable, _save_report, _virtualenv_active, cleanup_gitkeep_placeholders, configure_headless_matplotlib, configure_logging, current_session_log_path, detect_device, ensure_runtime_dependencies, ensure_runtime_directories, inspect_required_datasets, logger, platform, provision_required_datasets, sys
    ensure_runtime_directories()
    configure_logging()

    repair = ensure_runtime_dependencies(
        verbose=verbose,
    )

    loading = (
        LoadingBar(
            "Preparando chequeos de preboot"
        ).start()
        if verbose
        else None
    )
    try:
        if loading is not None:
            loading.update("Limpiando workspace local")
        removed = cleanup_gitkeep_placeholders()

        if loading is not None:
            loading.update("Verificando Python, configuracion y permisos")
        python_ok = sys.version_info >= MIN_PYTHON
        configs_ok = _configs_available()
        writable_ok = _runtime_writable()

        if loading is not None:
            loading.update("Comprobando dependencias instaladas")
        core = _dependency_status(CORE_DEPENDENCIES)
        ml = _dependency_status(ML_DEPENDENCIES)
        imagery = _dependency_status(
            IMAGERY_DEPENDENCIES
        )
        dev = _dependency_status(
            DEV_DEPENDENCIES
        )

        if loading is not None:
            loading.update("Detectando CPU y aceleradores GPU")
        device = detect_device(prefer_gpu=True)
        virtualenv_active = _virtualenv_active()

        if loading is not None:
            loading.update("Configurando visualizacion")
        visualization_backend = configure_headless_matplotlib()
    except Exception:
        if loading is not None:
            loading.fail("Fallo durante los chequeos de preboot")
        raise
    else:
        if loading is not None:
            loading.succeed("Chequeos iniciales completados")

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
        if repair.attempted:
            print(
                "  Dependencias       "
                f"[{'REPARADAS' if repair.success else 'ERROR'}]"
            )
        _print_group("Core", core)
        _print_group("ML completo", ml)
        _print_group("Imágenes", imagery)
        _print_group("Desarrollo", dev)
        print(
            f"  Acelerador         "
            f"{device.accelerator.upper()} — {device.name}"
        )
        if device.nvidia_device_names:
            print(
                "  NVIDIA sistema      "
                + ", ".join(device.nvidia_device_names)
            )
        print(
            "  XGBoost CUDA       "
            f"[{'SI' if device.xgboost_cuda_available else 'NO'}]"
        )
        print(
            f"  Diagnostico GPU    {device.diagnostic}"
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
        accelerator_diagnostic=device.diagnostic,
        nvidia_driver_available=device.nvidia_driver_available,
        nvidia_device_names=device.nvidia_device_names,
        xgboost_cuda_available=device.xgboost_cuda_available,
        virtualenv_active=virtualenv_active,
        visualization_backend=visualization_backend,
        dependency_repair=repair.to_dict(),
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
