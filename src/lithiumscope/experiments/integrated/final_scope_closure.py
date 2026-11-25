"""Summarize existing scientific simulation artifacts without modifying tracked documents."""
from pathlib import Path
import json
import pandas as pd
from lithiumscope.core.paths import RESULTS_DIR
from lithiumscope.core.hashing import file_sha256
from lithiumscope.results.atomic_exports import write_table,write_legacy_evidence_json,write_text

CAMPAIGNS=(
    ("2023-2025","batch_m2/batch_m2.csv"),
    ("2025 seasonal/shifted","campaign_2025/batch/batch_m2.csv"),
    ("2024 seasonal","seasonal_2024/batch/batch_m2.csv"),
    ("2024-2025 multiscene","temporal_multiscene/batch/batch_m2.csv"),
    ("same-scene spatial","controlled_spatial/all_684_scenario_metrics.csv"),
)


def summarize(base):
    base=Path(base)
    m1_path=base/"batch_m1/batch_m1.csv"
    m1=pd.read_csv(m1_path)
    rows=[]
    hashes={str(m1_path.relative_to(base)):file_sha256(m1_path)}
    for name,relative in CAMPAIGNS:
        path=base/relative
        frame=pd.read_csv(path)
        rows.append({"campaign":name,"total":len(frame),"scored":int(frame.status.eq("SCORED").sum()),
                     "expected_abstained":int(frame.status.eq("ABSTAIN_ALL_BAND_NODATA").sum()),
                     "unexpected":int((~frame.status.isin(["SCORED","ABSTAIN_ALL_BAND_NODATA"])).sum())})
        hashes[relative]=file_sha256(path)
    def read(relative):
        path=base/relative
        hashes[relative]=file_sha256(path)
        return json.loads(path.read_text(encoding="utf-8"))
    leak=read("leakage_safe_m2/experiment_manifest.json")
    spatial=read("controlled_spatial/spatial_robustness.json")
    historical=read("academic_evidence/recovered_training_validation.json")
    report=pd.DataFrame(rows)
    counts={"m1_scenarios":len(m1),"m2_scenarios":int(report.total.sum()),
            "m2_scored":int(report.scored.sum()),"m2_expected_abstained":int(report.expected_abstained.sum()),
            "m2_unexpected":int(report.unexpected.sum()),"spatial":spatial,"corrected_cv":leak,
            "historical":{k:historical[k] for k in ("dataset_hash_matches","label_match","split_fold_sizes_match",
                "split_group_overlaps","metrics_recomputed","fold_means","baseline_fold_means")},
            "external_accuracy":"NOT_EVALUATED","files_sha256":hashes}
    counts["historical_scope_consistent"]=(counts["m2_unexpected"]==0 and spatial.get("scored_patches")==36 and leak.get("rows")==706)
    counts["state"]="summarized" if counts["historical_scope_consistent"] else "review_required"
    out=base/"final_scope_closure"
    write_table(out/"m2_campaign_inventory.csv",report)
    write_legacy_evidence_json(out/"closure_manifest.json",counts)
    lines=["# LithiumScope — síntesis de simulaciones históricas","",
           "Resultados descriptivos recuperados; no constituyen validación externa de litio.","",
           "| Campaña | Total | Puntuadas | Abstenciones previstas | Inesperadas |",
           "|---|---:|---:|---:|---:|"]
    lines.extend("| {campaign} | {total} | {scored} | {expected_abstained} | {unexpected} |".format(**row) for row in rows)
    lines.extend(["",f"M1: {len(m1)} escenarios. M2: {counts['m2_scenarios']} escenarios.",
                  f"Estado: {counts['state']}.","",
                  "Las diferencias entre protocolos históricos no prueban superioridad ni aíslan efectos metodológicos.",
                  "Consultar closure_manifest.json para métricas recuperadas, diagnósticos y hashes de origen."])
    write_text(out/"SCIENTIFIC_SIMULATION_FINAL_CLOSURE.md","\n".join(lines)+"\n")
    return counts


def main():
    from lithiumscope.core.logging.session import run_log
    with run_log("simulations","final_scope_closure"):
        print(summarize(RESULTS_DIR/"scientific_simulations"))


if __name__=="__main__":
    main()
