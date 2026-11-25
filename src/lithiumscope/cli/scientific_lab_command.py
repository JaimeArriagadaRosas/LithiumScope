"""Interactive diagnostic laboratory; no fitting or promotion."""
from __future__ import annotations

import json
import os
import subprocess
import sys

from lithiumscope.cli.display import clear_screen, pause
from lithiumscope.cli.prompts import choose
from lithiumscope.core.paths import PROJECT_ROOT


def _prepare_eda(group):
    from lithiumscope.core.config import load_config
    from lithiumscope.experiments.generalization.prepared import prepare_version
    from lithiumscope.experiments.generalization.preflight import prepare_treatments, show_treatments
    snapshot, evidence = prepare_version(group, _eda_source(group))
    directory, manifest = prepare_treatments(group, snapshot, load_config(group))
    show_treatments(directory, manifest)
    from lithiumscope.cli.eda_verification import collect
    return evidence, manifest["state"], collect(snapshot, evidence, directory, manifest)


def _run_eda_m1():
    return _prepare_eda("model_1")


def _run_eda_m2():
    return _prepare_eda("model_2")


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
    records = {}
    for label, job in selected:
        try:
            from lithiumscope.core.logging.session import run_log
            with run_log("eda",label):
                outcome = job()
                path, prepared_state, record = outcome if isinstance(outcome, tuple) else (outcome, "complete", {})
                from lithiumscope.cli.eda_console import show_eda_summary
                displayed = show_eda_summary(path)
            status = {"complete": "COMPLETADO", "partial": "PARCIAL", "blocked": "BLOQUEADO"}[prepared_state]
            if not displayed and status == "COMPLETADO":
                status = "PARCIAL"
            states.append(status)
            records[label] = {"state":status,"console_summary_ok":displayed,**record}
            print(f"[{status}] {label}: {path}")
        except FileNotFoundError as exc:
            states.append("BLOQUEADO")
            records[label] = {"state":"BLOQUEADO","reason":str(exc)}
            print(f"[BLOQUEADO] {label}: {exc}")
        except Exception as exc:
            states.append("ERROR")
            records[label] = {"state":"ERROR","reason":str(exc)}
            print(f"[ERROR] {label}: {exc}")
    state = "COMPLETADO" if all(s == "COMPLETADO" for s in states) else "PARCIAL" if any(s in {"COMPLETADO","PARCIAL"} for s in states) else "ERROR" if "ERROR" in states else "BLOQUEADO"
    print(f"Estado: {state}. Diagnóstico y preparación experimental SOLO TRAIN; no se entrenaron modelos.")
    from lithiumscope.cli.eda_verification import write_verification
    print(f"Informe único para revisión: {write_verification(state, records)}")


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


def _treatment_experiments() -> bool:
    from lithiumscope.experiments.generalization.screen import run_screen
    from lithiumscope.core.logging.session import run_log
    from lithiumscope.results.atomic_exports import write_json
    from lithiumscope.core.paths import RESULTS_DIR
    from uuid import uuid4
    print("Evaluación espacial anidada: ajustes diagnósticos, no entrenamiento definitivo ni promoción.")
    print("Por defecto piloto Random Forest; alcance completo incluye las doce familias.")
    all_families=choose("¿Evaluar todas las familias? [s/n]: ",{"s","n"})=="s"
    records=[]
    for group in ("model_1","model_2"):
        try:
            with run_log("simulations",group+"_treatments"):
                path=run_screen(group,_eda_source(group),all_families=all_families)
            manifest=json.loads((path/"manifest.json").read_text(encoding="utf-8"))
            records.append({"model":group,"path":str(path),**manifest})
        except Exception as exc:
            records.append({"model":group,"state":"blocked" if isinstance(exc,FileNotFoundError) else "error","reason":str(exc)})
            print(f"[{records[-1]['state']}] {group}: {exc}")
    complete=all(r["state"]=="complete" for r in records)
    path=write_json(RESULTS_DIR/"scientific_simulations"/f"treatment_verification_{uuid4().hex[:12]}.json",
                    {"state":"COMPLETADO" if complete else "PARCIAL/BLOQUEADO","models":records,"promotion":False})
    print(f"Informe de tratamientos: {path}")
    return complete


def _validation_menu() -> bool | None:
    print("\nEJECUCIÓN DE TESTS Y SIMULACIONES")
    print("1. Tests completos del software")
    print("2. Simulaciones y comparación de modelos guardados")
    print("3. Evaluación experimental de tratamientos de datos")
    print("4. Ejecutar todo (tests + comparaciones + experimentación)")
    print("0. Volver")
    choice=choose("Seleccione una opción [0-4]: ",set("01234"))
    if choice=="0":
        return False
    operations=[_run_tests,_run_simulations,_treatment_experiments]
    selected=operations if choice=="4" else [operations[int(choice)-1]]
    outcomes=[]
    for operation in selected:
        try:
            outcomes.append(operation())
        except Exception as exc:
            print(f"[ERROR] {exc}")
            outcomes.append(False)
    if len(outcomes)>1:
        print("Estado global: "+("COMPLETADO" if all(outcomes) else "PARCIAL" if any(outcomes) else "ERROR/BLOQUEADO"))


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


def run() -> None:
    while True:
        clear_screen()
        print("\nLABORATORIO DE DIAGNÓSTICO")
        print("1. Análisis exploratorio")
        print("2. Ejecución de tests y simulaciones")
        print("3. Salud y diagnóstico del entorno")
        print("0. Volver")
        choice = choose("Seleccione una opción [0-3]: ", set("0123"))
        if choice == "0":
            return
        clear_screen()
        actions = {"1": _eda_menu, "2": _validation_menu, "3": _health}
        if actions[choice]() is not False:
            pause()
