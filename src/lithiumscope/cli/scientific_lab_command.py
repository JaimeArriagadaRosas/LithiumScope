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
    from lithiumscope.experiments.data_lab.workflow import analyze
    return analyze(group, _eda_source(group))


def _run_eda_m1():
    return _prepare_eda("model_1")


def _run_eda_m2():
    return _prepare_eda("model_2")


def _eda_source(model_group):
    from lithiumscope.datasets.management.sources.diagnostic import prepare_diagnostic_source
    try:
        return prepare_diagnostic_source(model_group)
    except FileNotFoundError:
        print(f"{model_group}: faltan datos locales. Las cohortes congeladas no se reconstruyen.")
        if choose("¿Autorizar adquisición de fuentes configuradas? [s/n]: ", {"s","n"}) != "s":
            raise
        from lithiumscope.runtime.dataset_provisioning import require_model_1_dataset, require_model_2_dataset
        def acquire_source(group):
            return (require_model_1_dataset if group == "model_1" else require_model_2_dataset)()
        return prepare_diagnostic_source(model_group, acquire=True, acquire_source=acquire_source)


def _eda_menu() -> bool | None:
    print("\nANÁLISIS EXPLORATORIO")
    print("1. Modelo 1")
    print("2. Modelo 2")
    print("3. Ambos modelos")
    print("0. Volver")
    choice = choose("Seleccione una opción [0-3]: ", set("0123"))
    if choice == "0":
        return False
    clear_screen()
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
    print(f"Estado: {state}. Análisis descriptivo; no se prepararon tratamientos ni se entrenaron modelos.")
    from lithiumscope.cli.eda_verification import write_verification
    print(f"Informe único para revisión: {write_verification(state, records, scope='descriptive_eda_only')}")


def _run_tests() -> bool:
    runner = PROJECT_ROOT / "scripts" / "run_tests.py"
    if not runner.is_file() or not (PROJECT_ROOT / "tests").is_dir():
        print("[BLOQUEADO] Suite local no disponible.")
        return False
    print("\nValidación del software | Comprobaciones estáticas, suite completa y cobertura", flush=True)
    from lithiumscope.core.logging.session import current_session_directory
    env={**os.environ,"LITHIUMSCOPE_SESSION_DIR":str(current_session_directory())}
    completed = subprocess.run([sys.executable, str(runner), "--preflight", "--coverage", "--", "-q", "--continue-on-collection-errors", "tests"], cwd=PROJECT_ROOT, check=False,env=env)
    return completed.returncode == 0


def _treatment_experiments(*, all_families=None, groups=("model_1", "model_2")) -> bool:
    from lithiumscope.experiments.generalization.screen import run_screen
    from lithiumscope.core.logging.session import run_log
    from lithiumscope.results.atomic_exports import write_json
    from lithiumscope.core.paths import RESULTS_DIR
    from uuid import uuid4
    print("EVALUACIÓN EXPERIMENTAL DE TRATAMIENTO DE DATOS\n")
    print("s: evaluación completa | n: piloto Random Forest")
    if all_families is None:
        selection=choose("¿Evaluar todas las familias? [s/n/0 cancelar]: ",{"s","n","0"})
        if selection=="0":
            return False
        all_families=selection=="s"
    print("\nModo: "+("evaluación completa" if all_families else "piloto Random Forest"))
    print("Modelos: " + ", ".join(group.replace("model_", "M") for group in groups))
    print("Algoritmos: "+("12 familias" if all_families else "Random Forest en ambos modelos"))
    print("Ajustes diagnósticos, sin guardar ni promover modelos.\n",flush=True)
    records=[]
    for group in groups:
        try:
            with run_log("simulations",group+"_treatments"):
                path=run_screen(group,_eda_source(group),all_families=all_families)
            manifest=json.loads((path/"manifest.json").read_text(encoding="utf-8"))
            import pandas as pd
            summary=path/"summary.csv"
            outer=path/"outer.csv"
            records.append({"model":group,"path":str(path),**manifest,
                "summary":json.loads(pd.read_csv(summary).to_json(orient="records")) if summary.is_file() else [],
                "outer":json.loads(pd.read_csv(outer).drop(columns=["features","feature_names"],errors="ignore").to_json(orient="records")) if outer.is_file() else []})
        except Exception as exc:
            records.append({"model":group,"state":"blocked" if isinstance(exc,FileNotFoundError) else "error","reason":str(exc)})
            print(f"[{records[-1]['state']}] {group}: {exc}")
    complete=all(r["state"]=="complete" for r in records)
    path=write_json(RESULTS_DIR/"scientific_simulations"/f"treatment_verification_{uuid4().hex[:12]}.json",
                    {"state":"COMPLETADO" if complete else "PARCIAL/BLOQUEADO","models":records,"promotion":False})
    print(f"Informe de tratamientos: {path}")
    return complete


def _select_models(title):
    print(f"\n{title}")
    print("1. Modelo 1")
    print("2. Modelo 2")
    print("3. Ambos modelos")
    print("0. Volver")
    choice = choose("Seleccione una opción [0-3]: ", set("0123"))
    return {"1": ("model_1",), "2": ("model_2",),
            "3": ("model_1", "model_2")}.get(choice, ())


def _comparison_menu():
    groups = _select_models("COMPARACIONES DE MODELOS")
    if not groups:
        return False
    clear_screen()
    from lithiumscope.cli.model_comparison_command import run
    return run(groups)


def _treatment_menu():
    groups = _select_models("EVALUACIÓN DE TRATAMIENTO DE DATOS")
    if not groups:
        return False
    clear_screen()
    if groups == ("model_1",):
        mode = choose("¿Evaluación general [g] o focalizada M1 [f]? [g/f/0]: ", {"g", "f", "0"})
        if mode == "0":
            return False
        if mode == "f":
            from lithiumscope.cli.m1_targeted_command import run
            return run()
    return _treatment_experiments(groups=groups)


def _validation_menu() -> bool | None:
    print("\nEJECUCIÓN DE TESTS Y SIMULACIONES\n")
    print("1. Validación del software")
    print("2. Comparaciones de modelos")
    print("3. Evaluación de tratamiento de datos")
    print("0. Volver")
    choice = choose("Seleccione una opción [0-3]: ", set("0123"))
    if choice == "0":
        return False
    clear_screen()
    try:
        return {"1": _run_tests, "2": _comparison_menu, "3": _treatment_menu}[choice]()
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return None


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


def _preparation_menu():
    from lithiumscope.cli.data_preparation_command import run
    return run()


def run() -> None:
    while True:
        clear_screen()
        print("\nLABORATORIO DE DIAGNÓSTICO")
        print("1. Análisis exploratorio")
        print("2. Preparación y tratamiento de datos")
        print("3. Ejecución de tests y simulaciones")
        print("4. Salud y diagnóstico del entorno")
        print("0. Volver")
        choice = choose("Seleccione una opción [0-4]: ", set("01234"))
        if choice == "0":
            return
        clear_screen()
        actions = {"1": _eda_menu, "2": _preparation_menu, "3": _validation_menu, "4": _health}
        if actions[choice]() is not False:
            pause()
