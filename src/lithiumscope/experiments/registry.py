"""Registered, inference-only simulations using local prerequisites."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from uuid import uuid4

import joblib
import pandas as pd

from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.paths import PROJECT_ROOT, RESULTS_DIR
from lithiumscope.persistence.active_models import active_model_path
from lithiumscope.experiments.scenarios.scenarios import evaluate_model_1
from lithiumscope.experiments.scenarios.batch import run_m2_batch, summarize
from lithiumscope.experiments.reporting.reporting import compare_geographies


def _active(group: str) -> tuple[dict, Path]:
    path = active_model_path(group)
    if path is None:
        raise FileNotFoundError(f"No hay modelo activo registrado para {group}.")
    return joblib.load(path), path


def run_registered_simulations(output_root: Path | None = None) -> Path:
    root = output_root if output_root is not None else RESULTS_DIR / "scientific_simulations"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    destination = root / f"sim_{stamp}_{uuid4().hex[:8]}"
    destination.mkdir(parents=True, exist_ok=False)
    protocols = []
    m2_batch = None

    def record(name, action):
        try:
            evidence = action()
            protocols.append({"protocol": name, "state": "COMPLETADO", **evidence})
        except FileNotFoundError as exc:
            protocols.append({"protocol": name, "state": "BLOQUEADO", "reason": str(exc)})
        except Exception as exc:
            protocols.append({"protocol": name, "state": "ERROR", "reason": str(exc)})
        print(f"{name}: {protocols[-1]['state']}")

    def m1():
        bundle, path = _active("model_1")
        source = PROJECT_ROOT / "examples" / "integrated_demo" / "cases.csv"
        cases = pd.read_csv(source)
        output = evaluate_model_1(bundle, cases, destination / "model_1")
        return {"cases": len(cases), "scenario_rows": len(pd.read_csv(output)),
                "model_sha256": file_sha256(path), "artifact": str(output)}

    def m2():
        nonlocal m2_batch
        bundle, path = _active("model_2")
        source = root / "sentinel2_real" / "manifest.csv"
        if not source.is_file():
            raise FileNotFoundError("Falta el catálogo de parches piloto; adquirirlo mediante protocolo explícito.")
        m2_batch = run_m2_batch(bundle, source, destination / "model_2")
        result = pd.read_csv(m2_batch)
        if result.empty:
            m2_batch = None
            raise FileNotFoundError("No hay imágenes piloto aceptadas.")
        summary = summarize(m2_batch, m2_batch.parent)
        return {"cases": int(result["image_id"].nunique()), "scenario_rows": len(result),
                "model_sha256": file_sha256(path), "artifact": str(m2_batch), "summary": str(summary)}

    def geography():
        if m2_batch is None:
            raise FileNotFoundError("No existen resultados M2 compatibles de esta sesión.")
        table, report = compare_geographies(m2_batch, destination / "geography")
        return {"cases": len(pd.read_csv(table)), "artifact": str(table), "summary": str(report)}

    record("M1: escenarios geoquímicos", m1)
    record("M2: perturbaciones espectrales", m2)
    record("M2: comparación geográfica y temporal", geography)
    states = {item["state"] for item in protocols}
    state = "COMPLETADO" if states == {"COMPLETADO"} else "PARCIAL" if "COMPLETADO" in states else "ERROR" if "ERROR" in states else "BLOQUEADO"
    manifest = destination / "manifest.json"
    manifest.write_text(json.dumps({"run_id": destination.name, "state": state,
                                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                                    "scope": "inference-only simulation; not independent validation",
                                    "protocols": protocols}, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest
