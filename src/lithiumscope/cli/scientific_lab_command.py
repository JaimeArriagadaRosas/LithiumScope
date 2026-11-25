"""Interactive diagnostic laboratory; no fitting or promotion."""
from __future__ import annotations

import json
import os
import subprocess
import sys

from lithiumscope.cli.display import clear_screen, pause
from lithiumscope.cli.prompts import choose
from lithiumscope.core.paths import PROJECT_ROOT


def _run_eda_m1():
    from lithiumscope.model_1.eda.pipeline import run_eda
    return run_eda(_eda_source("model_1"))


def _run_eda_m2():
    from lithiumscope.model_2.eda.pipeline import run_eda
    return run_eda(_eda_source("model_2"))


def _eda_source(model_group):
    from lithiumscope.datasets.management.diagnostic_provisioning import prepare_diagnostic_source
    try:
        return prepare_diagnostic_source(model_group)
    except FileNotFoundError:
        print(f"{model_group}: faltan datos locales. Las cohortes congeladas no se reconstruyen.")
        if choose("¿Autorizar adquisición de fuentes configuradas? [s/n]: ", {"s","n"}) != "s":
            raise
        return prepare_diagnostic_source(model_group,acquire=True)


def _eda_menu() -> bool | None:
    print("\nANÁLISIS EXPLORATORIO")
    print("1. Modelo 1")
    print("2. Modelo 2")
    print("3. Ambos modelos")
    print("0. Volver")
    choice = choose("Seleccione una opción [0-3]: ", set("0123"))
    if choice == "0":
        return False
    jobs = [("M1", _run_eda_m1), ("M2", _run_eda_m2)]
    selected = jobs if choice == "3" else [jobs[int(choice) - 1]]
    states = []
    for label, job in selected:
        try:
            from lithiumscope.core.logging.session import run_log
            with run_log("eda",label):
                path = job()
            states.append("COMPLETADO")
            print(f"[OK] {label}: {path}")
        except FileNotFoundError as exc:
            states.append("BLOQUEADO")
            print(f"[BLOQUEADO] {label}: {exc}")
        except Exception as exc:
            states.append("ERROR")
            print(f"[ERROR] {label}: {exc}")
    state = "COMPLETADO" if all(s == "COMPLETADO" for s in states) else "PARCIAL" if "COMPLETADO" in states else "ERROR" if "ERROR" in states else "BLOQUEADO"
    print(f"Estado: {state}. Alcance descriptivo; no se entrenaron modelos.")


def _run_tests() -> bool:
    runner = PROJECT_ROOT / "scripts" / "run_tests.py"
    if not runner.is_file() or not (PROJECT_ROOT / "tests").is_dir():
        print("[BLOQUEADO] Suite local no disponible.")
        return False
    print("[OK] Suite descubierta en tests/. Ejecutando; el resumen se obtiene de JUnit.")
    from lithiumscope.core.logging.session import current_session_directory
    env={**os.environ,"LITHIUMSCOPE_SESSION_DIR":str(current_session_directory())}
    completed = subprocess.run([sys.executable, str(runner), "--", "-q"], cwd=PROJECT_ROOT, check=False,env=env)
    print(f"Estado: {'COMPLETADO' if completed.returncode == 0 else 'ERROR'}; código={completed.returncode}")
    return completed.returncode == 0


def _run_simulations() -> bool:
    from lithiumscope.experiments.registry import run_registered_simulations

    from lithiumscope.core.logging.session import run_log
    with run_log("simulations","registered"):
        manifest = run_registered_simulations()
    record = json.loads(manifest.read_text(encoding="utf-8"))
    print(f"Estado: {record['state']}\nEvidencias: {manifest}")
    return record["state"] == "COMPLETADO"


def _validation_menu() -> bool | None:
    print("\nEJECUCIÓN DE TESTS Y SIMULACIONES")
    print("1. Ejecutar tests")
    print("2. Ejecutar simulaciones científicas")
    print("3. Ejecutar ambos")
    print("0. Volver")
    choice = choose("Seleccione una opción [0-3]: ", set("0123"))
    if choice == "0":
        return False
    operations = [_run_tests, _run_simulations]
    selected = operations if choice == "3" else [operations[int(choice) - 1]]
    outcomes = []
    for operation in selected:
        try:
            outcomes.append(operation())
        except Exception as exc:
            print(f"[ERROR] {exc}")
            outcomes.append(False)
    if len(outcomes) > 1:
        print("Estado global: " + ("COMPLETADO" if all(outcomes) else "PARCIAL" if any(outcomes) else "ERROR/BLOQUEADO"))


def _health() -> None:
    from lithiumscope.tools.inspection.health import write_health_report

    from lithiumscope.core.logging.session import run_log
    with run_log("diagnostics","health"):
        path = write_health_report()
    report = json.loads(path.read_text(encoding="utf-8"))
    print("\nSALUD Y DIAGNÓSTICO DEL ENTORNO\nModo: SOLO LECTURA")
    for area, result in report["areas"].items():
        print(f"{area:<35} {result['state']}")
    for recommendation in report["recommendations"]:
        print(f"- {recommendation}")
    print(f"Estado: {report['state']}\nInforme: {path}")
    print("[OK] No se aplicaron reparaciones.")


def _development_snapshot() -> None:
    from lithiumscope.tools.inspection.development.snapshot import run_interactive

    run_interactive()


def run() -> None:
    while True:
        clear_screen()
        print("\nLABORATORIO DE DIAGNÓSTICO")
        print("1. Análisis exploratorio")
        print("2. Ejecución de tests y simulaciones")
        print("3. Salud y diagnóstico del entorno")
        print("4. Evidencias de desarrollo (temporal)")
        print("0. Volver")
        choice = choose("Seleccione una opción [0-4]: ", set("01234"))
        if choice == "0":
            return
        clear_screen()
        actions = {"1": _eda_menu, "2": _validation_menu, "3": _health, "4": _development_snapshot}
        if actions[choice]() is not False:
            pause()
