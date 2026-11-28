"""M1 source-by-source execution, with no automatic downloads or model fits."""
from pathlib import Path
import json
from time import monotonic

import pandas as pd

from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.logging.session import get_logger
from lithiumscope.core.paths import DATA_DIR, RESULTS_DIR
from lithiumscope.datasets.management.sources.research import M1_RESEARCH_SOURCES
from lithiumscope.results.atomic_exports import write_json, write_table
from lithiumscope.results.eda_artifacts import new_run
from . import directed_profiles, historical_evidence, local_sources, source_catalog

# Historical scientific decisions remain context, not automatic new decisions.
NOTES = {
    "georoc_andean_202609": "Historical overlap; retrospective enrichment, no new rows approved.",
    "georoc_felsic_rocks": "Experimental transfer; stability guards not met.",
    "chen_global_arcs_2020": "Weighted transfer improved mean RMSE marginally but degraded high-Li tail.",
    "meixner_2020": "Insufficient predictor coverage and unresolved identity.",
    "fernandez_turiel_2020": "No compatible candidates in previous five-fold preflight.",
    "macusani_2025": "Quarantine: shared publication and potential duplicates.",
}


LABELS = {
    "georoc_andean_202609": "GEOROC Arco Andino",
    "georoc_felsic_rocks": "GEOROC Granitos y riolitas",
    "chen_global_arcs_2020": "Chen 2020",
    "meixner_2020": "Meixner 2020",
    "fernandez_turiel_2020": "Fernández-Turiel 2020",
    "macusani_2025": "Macusani 2025",
}
STATUS = {"complete":"COMPLETADO", "reused":"REUTILIZADO", "partial":"PARCIAL",
          "missing":"FALTANTE", "blocked":"BLOQUEADO"}


def comparison_row(record):
    """Compare descriptive denominators, not certified independent training samples."""
    summary = record.get("profile", {})
    key = record["source"]
    georoc = key.startswith("georoc_")
    qc = summary.get("qc_rows") if georoc else summary.get("basic_qc_rows")
    return {"source": key, "state": record["state"],
            "profile_rows": summary.get("candidate_rows" if georoc else "profile_rows"),
            "positive_li_rows": summary.get("positive_li_rows" if georoc else "uncensored_numeric_positive_li"),
            "positive_li_definition": ("harmonized_numeric_positive" if georoc else "uncensored_numeric_positive") if summary else None,
            "above_40_ppm": summary.get("tail_above_40"),
            "above_696_ppm": summary.get("tail_above_696"),
            "tail_definition": "numeric_li_in_profile_not_independence_or_exact_measurement" if summary else None,
            "qc_rows": qc,
            "qc_definition": ("oxide_total_94_102" if georoc else "source_specific_basic_qc") if qc is not None else None,
            "qc_no_identity_flag_rows": summary.get("qc_no_identity_flag_rows"),
            "above_696_qc_no_identity_flag": summary.get("above_696_qc_no_identity_flag"),
            "historical_evidence_verified": sum(v.get("state") == "verified_historical"
                                               for v in record.get("historical_evidence", [])),
            "approved_rows": record.get("approved_rows", 0),
            "scientific_disposition": record.get("scientific_disposition", "review_required"),
            "scope": "descriptive_not_independent_or_eligible",
            "profile_path": record.get("eda_path"),
            "historical_decision": record.get("historical_decision"),
            "reason": record.get("reason")}


def _show_source(record):
    row = comparison_row(record)
    print(f"  [{STATUS.get(record['state'], record['state'])}] {LABELS.get(record['source'], record['source'])}"
          f" | {record.get('elapsed_seconds', 0):.1f} s", flush=True)
    if record.get("profile"):
        for label, key in (("Filas perfiladas", "profile_rows"), ("Li positivo", "positive_li_rows"),
                           (">40 ppm", "above_40_ppm"), (">696 ppm", "above_696_ppm"),
                           ("QC", "qc_rows"), ("QC sin alerta de identidad", "qc_no_identity_flag_rows")):
            if row[key] is not None:
                print(f"    {label:<26} {row[key]:>8}")
        print(f"    Evidencias históricas verificadas: {row['historical_evidence_verified']}")
        print(f"    Nuevas filas aprobadas: {row['approved_rows']}")
    if record.get("reason"):
        print("    Incidencia: " + record["reason"])
    if record.get("eda_path"):
        print("    Perfil: " + Path(record["eda_path"]).name)


def _inspect(key, base_path, research_root, results_root):
    specs = {spec.key: spec for spec in M1_RESEARCH_SOURCES}
    if key in specs:
        from .georoc.candidate_eda import profile
        spec = specs[key]
        paths, inventory, receipt, plan_verified = local_sources.georoc_files(spec, research_root)
        complete = plan_verified and bool(paths) and all(v["state"] == "verified" for v in inventory)
        execute = lambda: profile(spec, paths, receipt, base_path,
            output_root=results_root / "model_1" / "candidate_eda" / key,
            availability_complete=complete, return_reuse=True)
    else:
        paths, inventory = local_sources.directed_files(directed_profiles.EXPECTED[key], research_root / "directed" / key)
        complete = bool(paths) and all(v["state"] == "verified" for v in inventory)
        execute = lambda: directed_profiles.profile(key, paths, base_path,
            output_root=results_root / "model_1" / "candidate_eda" / key)
    record = {"source": key, "files": inventory, "availability_complete": complete,
              "historical_decision": NOTES[key], "scientific_disposition": "review_required",
              "approved_rows": 0}
    verified = sum(item["state"] == "verified" for item in inventory)
    print(f"    Archivos verificados: {verified}/{len(inventory)}", flush=True)
    for item in inventory:
        if item["state"] != "verified":
            print(f"    {item['file']}: {item['state']} {item.get('reason','')}", flush=True)
    if not paths:
        record["state"] = "missing" if all(v["state"] == "missing" for v in inventory) else "blocked"
        return record
    try:
        path, reused = execute()
        manifest_path = path / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        record.update(eda_path=str(path), manifest_sha256=file_sha256(manifest_path),
                      profile=manifest.get("summary", {}), reused=reused)
        record["state"] = ("reused" if reused else "complete") if complete and base_path is not None else "partial"
    except Exception as exc:
        record.update(state="blocked", reason=f"{type(exc).__name__}: {exc}")
    return record


def run_sources(base_path, *, research_root=None, results_root=None, canonical_state="complete"):
    root = Path(results_root) if results_root is not None else RESULTS_DIR
    research_root = Path(research_root) if research_root is not None else DATA_DIR / "raw/model_1/research"
    catalog = root / "model_1" / "source_catalog"
    active, retired, blocked = [], [], {}
    for key in source_catalog.known_keys():
        try:
            record = source_catalog.retirement(key, root=catalog)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            blocked[key] = str(exc)
            record = None
        if record is None:
            active.append(key)
        else:
            retired.append({"source": key, "reason": record["reason"],
                            "decision_reference": record["decision_reference"]})
    records = []
    total = len(active) + 1
    print("\n" + "-"*72 + f"\nFUENTES ADICIONALES | {len(active)} activas\n" + "-"*72, flush=True)
    try:
        base_digest = file_sha256(base_path) if base_path is not None and Path(base_path).is_file() else None
    except OSError as exc:
        base_digest = None
        canonical_state = "blocked"
        print(f"[BLOQUEADO] Comparación canónica: {exc}", flush=True)
    if base_digest is None:
        base_path = None
    for index, key in enumerate(active, 2):
        print(f"\n[{index}/{total}] {LABELS.get(key,key)}", flush=True)
        started = monotonic()
        try:
            if key in blocked:
                raise ValueError("Retirement record requires review: " + blocked[key])
            record = _inspect(key, base_path, research_root, root)
        except Exception as exc:
            record = {"source": key, "state": "blocked", "reason": f"{type(exc).__name__}: {exc}"}
        if key not in blocked:
            try:
                record["historical_evidence"] = historical_evidence.inspect_history(key, root, base_digest)
            except Exception as exc:
                record["historical_evidence"] = [{"state": "blocked", "reason": str(exc)}]
        record["elapsed_seconds"] = round(monotonic()-started, 3)
        records.append(record)
        _show_source(record)
        get_logger("experiments.data_lab").info("M1 source=%s state=%s", key, record["state"])
    directory = new_run(catalog / "executions")
    complete = canonical_state == "complete" and all(r["state"] in ("complete", "reused") for r in records)
    payload = {"state": "complete" if complete else "partial", "canonical_state": canonical_state,
               "canonical_sha256": base_digest, "active": records, "retired": retired,
               "model_fits": 0, "new_rows_approved": 0}
    rows = [comparison_row(record) for record in records]
    table = pd.DataFrame(rows)
    comparison = directory / "source_comparison.csv"
    # Keep the comparison as an evidence artifact even when every source is retired.
    if table.empty:
        table = pd.DataFrame(columns=list(comparison_row({"source":"", "state":""})))
    write_table(comparison, table)
    payload["source_comparison"] = str(comparison)
    payload["source_comparison_sha256"] = file_sha256(comparison)
    path = write_json(directory / "source_inventory.json", payload)
    print("\n" + "="*72 + "\nRESUMEN DE FUENTES M1\n" + "="*72)
    print(f"Canónico: {STATUS.get(canonical_state, canonical_state)} | Activas: {len(active)} | Retiradas: {len(retired)}")
    print(f"{'Fuente':<29} {'Estado':<14} {'Perfil':>8} {'>40':>7} {'>696':>7}")
    for row in rows:
        cells = [str(row[k]) if row[k] is not None else "N/D" for k in ("profile_rows","above_40_ppm","above_696_ppm")]
        print(f"{LABELS.get(row['source'], row['source']):<29} {STATUS.get(row['state'],row['state']):<14} "
              f"{cells[0]:>8} {cells[1]:>7} {cells[2]:>7}")
    print("Conteos descriptivos; no representan nuevas muestras independientes.")
    print("Nuevas filas aprobadas: 0 | Ajustes: 0")
    print(f"Inventario: {path}\nComparación: {comparison}", flush=True)
    return records, retired, path
