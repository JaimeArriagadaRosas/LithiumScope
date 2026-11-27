"""Single interactive entry for saved comparisons and authorized robustness scenarios."""
from __future__ import annotations

import json

from lithiumscope.cli.prompts import choose


def run(groups):
    from lithiumscope.core.logging.session import run_log
    from lithiumscope.experiments.shared.version_comparison import run_version_comparison
    from lithiumscope.experiments.shared.registry import run_registered_simulations
    from lithiumscope.experiments.shared.robustness_evidence import find_reusable_robustness

    with run_log("simulations", "saved_version_comparison"):
        path = run_version_comparison(groups=groups)
    record = json.loads(path.read_text(encoding="utf-8"))
    print(f"Comparaciones guardadas: {record['state']} | {path}")
    for group, detail in record["models"].items():
        if detail["state"] != "complete":
            print(f"[PENDIENTE] {group}: {detail.get('reason', 'Evidencia no disponible')}")
    print("Las pruebas de robustez son complementarias; no reconstruyen OOF ni entrenan modelos.")
    pending = []
    for group in groups:
        cached, reason = find_reusable_robustness(group)
        if cached is not None:
            print(f"[REUTILIZADO] {group}: {cached}")
        else:
            print(f"[PENDIENTE] Robustez {group}: {reason}")
            pending.append(group)
    if not pending:
        return record["state"] == "complete"
    if choose("¿Autorizar ejecución de los escenarios pendientes? [s/n]: ", {"s", "n"}) != "s":
        print("Escenarios pendientes no ejecutados.")
        return record["state"] == "complete"
    outcomes = []
    for group in pending:
        try:
            with run_log("simulations", "registered_" + group):
                simulation = run_registered_simulations(groups=(group,), include_saved_oof=False)
            evidence = json.loads(simulation.read_text(encoding="utf-8"))
            print(f"Escenarios {group}: {evidence['state']} | {simulation}")
            outcomes.append(evidence["state"] == "COMPLETADO")
        except Exception as exc:
            print(f"[ERROR] Robustez {group}: {exc}")
            outcomes.append(False)
    return record["state"] == "complete" and all(outcomes)
