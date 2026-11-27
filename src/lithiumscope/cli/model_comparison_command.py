"""Single interactive entry for saved comparisons and authorized robustness scenarios."""
from __future__ import annotations

import json

from lithiumscope.cli.prompts import choose


def run(groups):
    from lithiumscope.core.logging.session import run_log
    from lithiumscope.experiments.shared.version_comparison import run_version_comparison
    from lithiumscope.experiments.shared.registry import run_registered_simulations

    with run_log("simulations", "saved_version_comparison"):
        path = run_version_comparison(groups=groups)
    record = json.loads(path.read_text(encoding="utf-8"))
    print(f"Comparaciones guardadas: {record['state']} | {path}")
    for group, detail in record["models"].items():
        if detail["state"] != "complete":
            print(f"[PENDIENTE] {group}: {detail.get('reason', 'Evidencia no disponible')}")
    print("Las pruebas de robustez son complementarias; no reconstruyen OOF ni entrenan modelos.")
    print("No se repiten escenarios automáticamente.")
    if choose("¿Autorizar una nueva ejecución de escenarios de robustez? [s/n]: ", {"s", "n"}) != "s":
        return record["state"] == "complete"
    with run_log("simulations", "registered"):
        simulation = run_registered_simulations(groups=groups, include_saved_oof=False)
    evidence = json.loads(simulation.read_text(encoding="utf-8"))
    print(f"Escenarios: {evidence['state']} | {simulation}")
    return record["state"] == "complete" and evidence["state"] == "COMPLETADO"
