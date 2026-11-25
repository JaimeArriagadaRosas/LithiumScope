"""Interactive launcher for experimental M1-only refinement."""
from __future__ import annotations

import json


def run():
    from lithiumscope.cli.prompts import choose
    from lithiumscope.core.logging.session import run_log
    from lithiumscope.experiments.model_1.targeted.protocol import available_runs
    from lithiumscope.experiments.model_1.targeted.runner import run_targeted

    runs = available_runs()
    if not runs:
        print("[BLOQUEADO] No hay competencias M1 con tablas y checkpoints locales.")
        return False
    runs = runs[:9]
    for i, run_id in enumerate(runs,1):
        print(f"{i}. {run_id}")
    print("0. Volver")
    choice = choose("Seleccione competencia de referencia: ", {"0", *(str(i) for i in range(1,len(runs)+1))})
    if choice == "0":
        return False
    print("45 configuraciones: RF, HGB y CatBoost; CV espacial anidada; sin guardar ni promover modelos.")
    if choose("¿Ejecutar refinamiento dirigido? [s/n]: ", {"s","n"}) != "s":
        return False
    with run_log("simulations","m1_targeted"):
        result = run_targeted(runs[int(choice)-1])
    return json.loads((result/"manifest.json").read_text(encoding="utf-8"))["state"] == "complete"
