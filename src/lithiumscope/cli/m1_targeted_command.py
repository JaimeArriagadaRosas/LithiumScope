"""Launch the approved M1 refinement campaign directly from the lab."""
from __future__ import annotations

import json

# Freeze the campaign's agreed reference, never silently choose another training.
REFERENCE_RUN = "training_20261125_115932_m0300_826123c1"


def run():
    from lithiumscope.core.logging.session import run_log
    from lithiumscope.experiments.model_1.targeted.protocol import available_runs
    from lithiumscope.experiments.model_1.targeted.runner import run_targeted

    print("EVALUACIÓN FOCALIZADA DE M1\n")
    print(f"Competencia de referencia: {REFERENCE_RUN}")
    print("45 configuraciones: RF, HGB y CatBoost; CV espacial anidada.")
    print("Sin guardar ni promover modelos. M2 no participa.\n", flush=True)
    if REFERENCE_RUN not in available_runs():
        print("[BLOQUEADO] Faltan tablas o checkpoints de la competencia de referencia.")
        print("No se sustituirá por otro entrenamiento.")
        return False
    with run_log("simulations", "m1_targeted"):
        result = run_targeted(REFERENCE_RUN)
    return json.loads((result/"manifest.json").read_text(encoding="utf-8"))["state"] == "complete"
