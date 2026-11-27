"""Expand external whole-rock provenance screening using original GEOROC rows."""
from pathlib import Path
from lithiumscope.experiments.data_lab.source_studies.georoc.protocol import EXPERIMENT_ROOT
import json,re,hashlib
from ..provenance.bibliography import unambiguous_references
import numpy as np
import pandas as pd

ROOT=EXPERIMENT_ROOT
SECTIONS=ROOT/"georoc_andean_raw"
CITATIONS=ROOT/"georoc_citations_2024.tab"
def run(root=ROOT, *, candidate_path=None, partitions=None, citations=CITATIONS):
    root=Path(root)
    candidate_path=Path(candidate_path) if candidate_path is not None else root/"georoc_provenance_screen.csv"
    candidates=pd.read_csv(candidate_path)
    refs=pd.read_csv(citations,sep="\t",encoding="latin1",low_memory=False)
    refs, ambiguous=unambiguous_references(refs)
    look=refs.set_index("citation_number")
    fields=[]
    for f in (sorted(map(Path,partitions)) if partitions is not None else sorted((root/"georoc_andean_raw").glob("*.csv"))):
        tab=pd.read_csv(f,encoding="latin1",low_memory=False,
          usecols=["SAMPLE NAME","LATITUDE MIN","LONGITUDE MIN","LI(PPM)","MATERIAL",
                   "UNIQUE_ID","DRILL DEPTH MIN","ROCK TYPE","LOCATION COMMENT"])
        tab["source_file"]=f.name
        fields.append(tab)
    if not fields:
        raise FileNotFoundError("No original GEOROC partitions")
    source=pd.concat(fields,ignore_index=True)
    source["lat"]=pd.to_numeric(source["LATITUDE MIN"],errors="coerce")
    source["lon"]=pd.to_numeric(source["LONGITUDE MIN"],errors="coerce")
    source["li"]=pd.to_numeric(source["LI(PPM)"],errors="coerce")
    keys=["source_file","SAMPLE NAME","lat","lon","li"]
    source["primary_match_count"]=source.groupby(keys,dropna=False)["source_file"].transform("size")
    # Keep all competing source rows separately; ambiguous rows never become eligible.
    source.loc[source.primary_match_count.gt(1)].to_csv(root/"ambiguous_primary_rows.csv",index=False)
    source=source.drop_duplicates(subset=keys,keep="first")
    merged=candidates.merge(source, left_on=["source_file","sample_name","latitude","longitude","li_ppm"],
        right_on=["source_file","SAMPLE NAME","lat","lon","li"],how="left",validate="many_to_one")
    merged["primary_row_verified"]=merged.primary_match_count.eq(1)&merged["UNIQUE_ID"].notna()
    merged["primary_match_count"]=merged.primary_match_count.fillna(0).astype(int)
    cite_numbers=merged.citation_key.fillna("").astype(str).str.replace(r"\.0$","",regex=True).str.split(";").map(lambda x:int(x[0]) if x and x[0].isdigit() else -1)
    merged["citation_number"]=cite_numbers
    merged["resolved_citation"]=cite_numbers.isin(look.index)
    merged["ambiguous_citation"]=cite_numbers.isin(ambiguous)
    merged["multiple_references"]=merged.citation_key.fillna("").astype(str).str.contains(";",regex=False)
    merged["citation_doi"]=cite_numbers.map(look["DOI"])
    merged["citation_title"]=cite_numbers.map(look["TITLE"])
    depth=pd.to_numeric(merged["DRILL DEPTH MIN"],errors="coerce")
    merged["drilled_below_surface"]=depth.gt(0)
    merged["plutonic_rock"]=merged["ROCK TYPE"].fillna("").str.upper().eq("PLU")
    merged["candidate_gate"]=np.select([
        ~merged.primary_row_verified,~np.isfinite(merged.nearest_training_km),
        merged.ambiguous_citation,merged.multiple_references,
        merged.nearest_training_km.lt(100),merged.duplicate_identifier_location,
        merged.drilled_below_surface,merged.plutonic_rock,~merged.resolved_citation],
        ["PRIMARY_ROW_UNRESOLVED","INVALID_DISTANCE","AMBIGUOUS_BIBLIOGRAPHY","MULTIPLE_REFERENCES_REVIEW",
         "NEAR_TRAINING","IDENTIFIER_DUPLICATE","BURIED","PLUTONIC","MISSING_BIBLIOGRAPHY"],
        default="REVIEW_METHOD_AND_SURFACE")
    merged["validation_approved"]=False
    merged.to_csv(root/"georoc_extended_evidence_screen.csv",index=False)
    overview={
        "candidate_total":len(merged),
        "gate_counts":merged.candidate_gate.value_counts().to_dict(),
        "resolved_bibliography_total":int(merged.resolved_citation.sum()),
        "drill_depth_positive_total":int(merged.drilled_below_surface.sum()),
        "primary_source_sha256":hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
        "approved_external_labels":0,
        "warning":"Bibliography resolved does not confirm element-specific analytic method or source-training independence."
    }
    (root/"georoc_extended_evidence_summary.json").write_text(json.dumps(overview,indent=2),encoding="utf8")
    print(json.dumps(overview,indent=2),flush=True)
    top=merged.loc[merged.candidate_gate.eq("REVIEW_METHOD_AND_SURFACE") &
        merged.country.eq("CHILE")].copy()
    top=top.drop_duplicates("citation_number").sort_values("nearest_training_km",ascending=False)
    top[["sample_name","citation_number","citation_doi","latitude","longitude","li_ppm","nearest_training_km"]].head(25).to_csv(
        root/"method_review_shortlist.csv",index=False)
    print("SHORTLIST",len(top),flush=True)
if __name__=="__main__":
    from lithiumscope.core.logging.session import run_log
    with run_log("simulations","georoc_extended_screen"):
        run()
