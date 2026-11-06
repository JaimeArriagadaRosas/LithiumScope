"""Audit each GEOROC pilot specimen against original row and published methods.

This is a safety/quality GATE, not an external-scoring pipeline.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path("results/scientific_simulations/external_geochemistry_audit")
EVIDENCE = {
    "27313": {
        "source": "https://doi.org/10.1016/j.oregeorev.2018.08.034",
        "method": "WHOLE_ROCK_ICP_MS_REPORTED_WITH_LI_BORATE_FUSION",
        "method_note": "Published chemistry mentions Li-bearing metaborate/tetraborate fusion; element-specific Li route unresolved.",
        "sample_match": "SAMPLE_NAME_IN_ORIGINAL_ARTICLE_TABLE",
    },
    "29268": {
        "source": "https://repositorio.udec.cl/server/api/core/bitstreams/def510a7-f45a-400c-acef-37ab5a1073a0/content",
        "method": "ICP_MS_AND_ICP_OES_GROUP_REPORTED",
        "method_note": "Gonzalez Alarcon (2023) thesis Table 3.2 reports CPV-16-429 Li 41.6 ppm; sample-specific elemental route requires Appendix 3.A.",
        "sample_match": "SAMPLE_AND_LI_VALUE_CONFIRMED_IN_THESIS",
    },
    "29484": {
        "source": "",
        "method": "UNKNOWN",
        "method_note": "Original citation unresolved. Source row records drilling depth 100.1 m.",
        "sample_match": "ONLY_GEOROC_ROW",
    },
    "23382": {
        "source": "https://eartharxiv.org/repository/object/1167/download/2644/",
        "method": "WHOLE_ROCK_TRACE_ELEMENT_ICP_MS",
        "method_note": "Wieser et al. appendix SIA3.1 describes whole-rock trace-element ICP-MS; Li-specific result still requires source-table audit.",
        "sample_match": "SAMPLE_ID_IN_PAPER_APPENDIX",
    },
}


def sample_surface_gate(depth: object, rock_type: str) -> str:
    value = pd.to_numeric(depth, errors="coerce")
    if pd.notna(value) and float(value) > 0:
        return "EXCLUDE_BURIED_DRILL_SAMPLE"
    if str(rock_type).upper().startswith("PLU"):
        return "HOLD_PLUTONIC_ENCLAVE_SURFACE_MATCH"
    return "HOLD_SURFACE_REPRESENTATIVENESS_UNVERIFIED"


def run(base: Path = BASE) -> tuple[Path, Path]:
    pilot = pd.read_csv(base / "pilot_target_metadata.csv")
    training = pd.read_csv(
        r"D:\Descargas\LithiumScope\data\processed\model_2\training_manifest.csv"
    )
    source_cache: dict[str, pd.DataFrame] = {}
    results = []
    for item in pilot.to_dict("records"):
        part = str(item["source_file"])
        if part not in source_cache:
            source_cache[part] = pd.read_csv(
                base / "georoc_andean_raw" / part, encoding="latin1", low_memory=False
            )
        source = source_cache[part]
        matched = source.loc[source["SAMPLE NAME"].eq(item["sample_name"])]
        if len(matched) != 1:
            raise ValueError(f"Expected exactly one primary row for {item['image_id']}: {len(matched)}")
        raw = matched.iloc[0]
        li = pd.to_numeric(raw["LI(PPM)"], errors="coerce")
        if not np.isfinite(li) or abs(li - float(item["li_ppm"])) > 1e-6:
            raise ValueError(f"Li provenance mismatch: {item['image_id']}")
        ident = str(item["image_id"]).removeprefix("georoc_")
        evidence = EVIDENCE[ident]
        depth = pd.to_numeric(raw.get("DRILL DEPTH MIN"), errors="coerce")
        gate = sample_surface_gate(depth, str(raw.get("ROCK TYPE", "")))
        results.append({
            "image_id": item["image_id"],
            "georoc_unique_id": str(raw["UNIQUE_ID"]),
            "sample_name": item["sample_name"],
            "source_csv": part,
            "citation": raw["CITATIONS"],
            "source_li_ppm": li,
            "matches_imported_li": True,
            "nearest_training_km": item["nearest_training_km"],
            "rock_type": raw.get("ROCK TYPE"),
            "material": raw.get("MATERIAL"),
            "location_comment": raw.get("LOCATION COMMENT"),
            "drill_depth_m": float(depth) if pd.notna(depth) else np.nan,
            "surface_gate": gate,
            "source_document": evidence["source"],
            "analytical_method_evidence": evidence["method"],
            "method_caveat": evidence["method_note"],
            "sample_source_match": evidence["sample_match"],
            "independence_bibliographic": "UNVERIFIED",
            "eligibility_for_external_accuracy": "NOT_APPROVED",
        })
    table = pd.DataFrame(results)
    outfile = base / "pilot_sample_method_surface_audit.csv"
    table.to_csv(outfile, index=False)
    sources = base / "primary_sources"
    payload = {
        "total": len(table),
        "buried_drill_samples_excluded": int(table.surface_gate.eq("EXCLUDE_BURIED_DRILL_SAMPLE").sum()),
        "samples_matched_to_primary_georoc_rows": int(table.matches_imported_li.sum()),
        "external_accuracy_labels_approved": 0,
        "source_pdf_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources.glob("*.pdf")
        },
        "caveats": [
            "Li-bearing fusion flux in the Del Real chemistry procedure requires element-level method review.",
            "CPV sample is plutonic enclave, not a guaranteed surface reflectance match.",
            "A 100.1m drill-core reference is not a surface-imagery label.",
            "Reference dataset and source-paper overlaps with Mamani09 require further provenance checks.",
        ],
    }
    report = base / "pilot_sample_method_surface_audit.json"
    report.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(table[["image_id", "surface_gate", "analytical_method_evidence",
                 "sample_source_match"]].to_string(index=False), flush=True)
    print(json.dumps(payload, indent=2), flush=True)
    return outfile, report


if __name__ == "__main__":
    run()
