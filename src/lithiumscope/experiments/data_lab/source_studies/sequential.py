"""M1 source-by-source execution, with no automatic downloads or model fits."""
from pathlib import Path
import json

from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.logging.session import get_logger
from lithiumscope.core.paths import DATA_DIR, RESULTS_DIR
from lithiumscope.datasets.management.sources.research import M1_RESEARCH_SOURCES
from lithiumscope.results.atomic_exports import write_json
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
    for item in inventory:
        print(f"    {item['file']}: {item['state']}", flush=True)
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
    print(f"[1/{total}] Dataset canónico: {canonical_state}", flush=True)
    base_digest = file_sha256(base_path) if base_path is not None and Path(base_path).is_file() else None
    if base_digest is None:
        base_path = None
    for index, key in enumerate(active, 2):
        print(f"\n[{index}/{total}] {key}", flush=True)
        try:
            if key in blocked:
                raise ValueError("Retirement record requires review: " + blocked[key])
            record = _inspect(key, base_path, research_root, root)
        except Exception as exc:
            record = {"source": key, "state": "blocked", "reason": f"{type(exc).__name__}: {exc}"}
        if key not in blocked:
            record["historical_evidence"] = historical_evidence.inspect_history(key, root, base_digest)
        records.append(record)
        print(f"  [{record['state'].upper()}] {key}: {record.get('profile', record.get('reason', ''))}", flush=True)
        get_logger("experiments.data_lab").info("M1 source=%s state=%s", key, record["state"])
    directory = new_run(catalog / "executions")
    complete = canonical_state == "complete" and all(r["state"] in ("complete", "reused") for r in records)
    payload = {"state": "complete" if complete else "partial", "canonical_state": canonical_state,
               "canonical_sha256": base_digest, "active": records, "retired": retired,
               "model_fits": 0, "new_rows_approved": 0}
    path = write_json(directory / "source_inventory.json", payload)
    print(f"\nFuentes activas: {len(active)} | retiradas: {len(retired)} | inventario: {path}", flush=True)
    for record in records:
        print(f"  {record['source']}: {record['state']}")
    return records, retired, path
