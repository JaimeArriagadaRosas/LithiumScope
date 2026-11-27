"""Reproduce bibliography verification and training/source overlap for GEOROC pilot."""
from pathlib import Path
from lithiumscope.experiments.data_lab.source_studies.georoc.protocol import EXPERIMENT_ROOT
import re,json,hashlib
import pandas as pd

ROOT=EXPERIMENT_ROOT
from lithiumscope.core.paths import PROJECT_ROOT
TRAIN=PROJECT_ROOT/"data/processed/model_2/training_manifest.csv"
SOURCE_DOC=PROJECT_ROOT / "docs/data/DATASETS_AND_FEATURES.md"


def unambiguous_references(refs):
    """Keep exact duplicate metadata but withhold conflicting citation identities."""
    table = refs.copy()
    table["citation_number"] = pd.to_numeric(table.CITATIONS, errors="coerce")
    table = table.loc[table.citation_number.notna()].drop_duplicates()
    ambiguous = set(table.loc[table.citation_number.duplicated(keep=False), "citation_number"])
    return table.loc[~table.citation_number.isin(ambiguous)], ambiguous


def main():
    ref_file=ROOT/"georoc_citations_2024.tab"
    refs=pd.read_csv(ref_file,sep="\t",encoding="latin1",low_memory=False)
    targets=pd.read_csv(ROOT/"pilot_target_metadata.csv")
    train=pd.read_csv(TRAIN)
    doc=SOURCE_DOC.read_text(encoding="utf8",errors="replace")
    refs, ambiguous = unambiguous_references(refs)
    ref_ids=refs.citation_number
    records=[]
    for row in targets.to_dict("records"):
        links=[]
        for ref in re.findall(r"\d+",str(row["citation_ids"])):
            matching=refs.loc[ref_ids.eq(int(ref))]
            if int(ref) in ambiguous:
                links.append({"reference_id":ref,"status":"UNRESOLVED_AMBIGUOUS_CITATION"})
            elif matching.empty:
                links.append({"reference_id":ref,"status":"UNRESOLVED_IN_2024_CITATION_TABLE"})
            else:
                item=matching.iloc[0]
                links.append({"reference_id":ref,"status":"BIBLIOGRAPHY_RESOLVED",
                    "doi":str(item.get("DOI","")) if pd.notna(item.get("DOI")) else None,
                    "title":str(item.get("TITLE","")),
                    "year":str(item.get("YEAR",""))})
        normalized=re.sub(r"\s*\[\d+\]","",row["sample_name"]).strip().upper()
        literal_training_overlap=normalized in set(train.sample_id.astype(str).str.upper())
        same_ref_as_explicit_training=any(
            (v.get("doi") and v["doi"].lower() in doc.lower())
            for v in links
        )
        records.append({
            "image_id":row["image_id"],"sample_name":row["sample_name"],"li_ppm":row["li_ppm"],
            "nearest_training_km":row["nearest_training_km"],
            "references":json.dumps(links,ensure_ascii=False),
            "unresolved_references":sum(v["status"].startswith("UNRESOLVED") for v in links),
            "literal_training_id_overlap":literal_training_overlap,
            "reference_doi_in_training_document":same_ref_as_explicit_training,
            "analytical_method":"UNVERIFIED",
            "bibliographic_independence":"UNVERIFIED",
            "approval":"NOT_APPROVED",
        })
    result=pd.DataFrame(records)
    result.to_csv(ROOT/"pilot_bibliography_audit.csv",index=False)
    summary={
        "pilot_count":len(records),
        "bibliographic_refs_resolved":sum(len(json.loads(r["references"]))-r["unresolved_references"] for r in records),
        "bibliographic_refs_unresolved":int(result.unresolved_references.sum()),
        "literal_training_sample_id_overlaps":int(result.literal_training_id_overlap.sum()),
        "approved_external_labels":0,
        "training_source_note":"M2 training manifest derives from Mamani09_Table_DR2 bootstrap mirror per local DATASETS_AND_FEATURES.md; dataset-level independence is not established by differing reference IDs.",
        "citation_file_sha256":hashlib.sha256(ref_file.read_bytes()).hexdigest(),
        "training_manifest_sha256":hashlib.sha256(TRAIN.read_bytes()).hexdigest(),
        "caution":"2024 citation table may not contain 2026 compilation additions; missing methods require source papers.",
    }
    (ROOT/"pilot_bibliography_manifest.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf8")
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    print(result[["image_id","unresolved_references","literal_training_id_overlap","approval"]].to_string(index=False),flush=True)

if __name__=="__main__":
    from lithiumscope.core.logging.session import run_log
    with run_log("simulations","georoc_bibliography_audit"):
        main()
