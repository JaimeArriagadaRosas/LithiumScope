"""Read-only consultation; candidate administration is not part of this menu."""
from __future__ import annotations

import json
import os
import webbrowser

from lithiumscope.cli.prompts import choose
from lithiumscope.core.paths import RESULTS_DIR
from lithiumscope.results.catalog import run_catalog
from lithiumscope.results.evidence import active_summary, document_identity, latest_document, latest_record


def _open_document(suffix: str) -> None:
    path = latest_document(RESULTS_DIR, suffix)
    if path is None:
        print(f"[INFORMACIÓN] No hay archivos {suffix} compatibles disponibles.")
        print("No se ha iniciado ninguna operación de generación.")
        return
    identity = document_identity(path, RESULTS_DIR)
    print(f"Ejecución: {identity.get('run_id', 'N/D')}")
    print(f"Tipo: {identity.get('mode', identity.get('model_group', 'N/D'))}")
    print(f"Archivo: {path}")
    try:
        if os.name == "nt":
            os.startfile(path)
            opened = True
        else:
            opened = webbrowser.open(path.resolve().as_uri())
        print("[OK] Solicitud de apertura enviada." if opened else "[ADVERTENCIA] No se pudo abrir el visor.")
    except OSError as exc:
        print(f"[ERROR] No se pudo abrir el visor: {exc}")
    print("No se ha generado un nuevo archivo.")


def _pdf() -> None:
    _open_document(".pdf")


def _excel() -> None:
    _open_document(".xlsx")


def _models() -> None:
    for group in ("model_1", "model_2"):
        print(f"\n{group.upper()} | Modelo activo")
        print(json.dumps(active_summary(group), ensure_ascii=False, indent=2))
        try:
            catalog = run_catalog(group)
            if not catalog.empty:
                fields = ["run_id", "winner", "state", "primary_metric", "primary_metric_value", "release_gate_pass"]
                print("Candidatos locales (métricas según su propio protocolo):")
                print(catalog[fields].fillna("N/D").head(10).to_string(index=False))
            else:
                print("Candidatos locales: N/D")
        except (ValueError, OSError, KeyError) as exc:
            print(f"[ADVERTENCIA] Catálogo no legible: {exc}")
    print("[ADVERTENCIA] Métricas de alcance distinto no demuestran superioridad directa.")


def _eda() -> None:
    for group in ("model_1", "model_2"):
        item = latest_record(RESULTS_DIR / group / "eda", "manifest.json")
        print(f"\n{group.upper()}")
        if item is None:
            print("N/D — no hay manifiesto EDA disponible.")
            continue
        path, record = item
        print(json.dumps(record, ensure_ascii=False, indent=2))
        print(f"Evidencia: {path}")
        print("Vigencia respecto del próximo entrenamiento: requiere comprobar la firma efectiva.")
    print("No se ejecutó un nuevo análisis exploratorio.")


def _tests() -> None:
    item = latest_record(RESULTS_DIR / "tests", "summary.json")
    if item:
        path, record = item
        print("TESTS")
        print(json.dumps(record, ensure_ascii=False, indent=2))
        print(f"Evidencia: {path}")
        for name in ("verification_report.md", "architecture.json", "criterion_evidence.json"):
            evidence = path.parent / name
            if evidence.is_file():
                print(f"{name}: {evidence}")
    else:
        print("Tests: N/D — no hay resumen registrado.")
    root = RESULTS_DIR / "scientific_simulations"
    for name in ("manifest.json", "experiment_manifest.json", "closure_manifest.json"):
        item = latest_record(root, name)
        if item:
            print(f"Simulación registrada: {item[0]}")
    print("No se ejecutaron nuevos tests ni simulaciones.")


def run() -> bool | None:
    print("\nMÉTRICAS Y RESULTADOS")
    print("1. Visualizar PDF")
    print("2. Visualizar Excel")
    print("3. Métricas y estado de los modelos")
    print("4. Resultados del análisis exploratorio")
    print("5. Resultados de tests y simulaciones")
    print("0. Volver")
    choice = choose("Seleccione una opción [0-5]: ", set("012345"))
    if choice == "0":
        return False
    actions = {"1": _pdf, "2": _excel, "3": _models, "4": _eda, "5": _tests}
    actions[choice]()
