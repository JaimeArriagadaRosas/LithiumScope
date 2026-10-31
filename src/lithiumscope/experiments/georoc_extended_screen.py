"""Expand external whole-rock provenance screening using original GEOROC rows."""
from pathlib import Path
import json,re,hashlib
import numpy as np
import pandas as pd

ROOT=Path("results/scientific_simulations/external_geochemistry_audit")
SECTIONS=ROOT/"georoc_andean_raw"
CITATIONS=ROOT/"georoc_citations_2024.tab"
def run():
    candidates=pd.read_csv(ROOT/"georoc_provenance_screen.csv")
    refs=pd.read_csv(CITATIONS,sep="\t",encoding="latin1",low_memory=False)
    refs["citation_number"]=pd.to_numeric(refs.CITATIONS,errors="coerce").astype("Int64")
    look=refs.drop_duplicates("citation_number").set_index("citation_number")
    fields=[]
    for f in sorted(SECTIONS.glob("*.csv")):
        tab=pd.read_csv(f,encoding="latin1",low_memory=False,
          usecols=["SAMPLE NAME","LATITUDE MIN","LONGITUDE MIN","LI(PPM)","MATERIAL",
                   "UNIQUE_ID","DRILL DEPTH MIN","ROCK TYPE","LOCATION COMMENT"])
        tab["source_file"]=f.name
        fields.append(tab)
    source=pd.concat(fields,ignore_index=True)
    source["lat"]=pd.to_numeric(source["LATITUDE MIN"],errors="coerce")
    source["lon"]=pd.to_numeric(source["LONGITUDE MIN"],errors="coerce")
    source["li"]=pd.to_numeric(source["LI(PPM)"],errors="coerce")
    source=source.drop_duplicates(subset=["source_file","SAMPLE NAME","lat","lon","li"],keep="first")
    merged=candidates.merge(source, left_on=["source_file","sample_name","latitude","longitude","li_ppm"],
        right_on=["source_file","SAMPLE NAME","lat","lon","li"],how="left",validate="one_to_one")
    if merged["UNIQUE_ID"].isna().sum():raise ValueError("Could not match source rows")
    cite_numbers=merged.citation_key.fillna("").str.split(";").map(lambda x:int(x[0]) if x and x[0].isdigit() else -1)
    merged["citation_number"]=cite_numbers
    merged["resolved_citation"]=cite_numbers.isin(look.index)
    merged["citation_doi"]=cite_numbers.map(look["DOI"])
    merged["citation_title"]=cite_numbers.map(look["TITLE"])
    depth=pd.to_numeric(merged["DRILL DEPTH MIN"],errors="coerce")
    merged["drilled_below_surface"]=depth.gt(0)
    merged["plutonic_rock"]=merged["ROCK TYPE"].fillna("").str.upper().eq("PLU")
    merged["candidate_gate"]=np.select([
        merged.nearest_training_km.lt(100),merged.duplicate_identifier_location,
        merged.drilled_below_surface,merged.plutonic_rock,~merged.resolved_citation],
        ["NEAR_TRAINING","IDENTIFIER_DUPLICATE","BURIED","PLUTONIC","MISSING_BIBLIOGRAPHY"],
        default="REVIEW_METHOD_AND_SURFACE")
    merged["validation_approved"]=False
    merged.to_csv(ROOT/"georoc_extended_evidence_screen.csv",index=False)
    overview={
        "candidate_total":len(merged),
        "gate_counts":merged.candidate_gate.value_counts().to_dict(),
        "resolved_bibliography_total":int(merged.resolved_citation.sum()),
        "drill_depth_positive_total":int(merged.drilled_below_surface.sum()),
        "primary_source_sha256":hashlib.sha256((ROOT/"georoc_andean_candidates.csv").read_bytes()).hexdigest(),
        "approved_external_labels":0,
        "warning":"Bibliography resolved does not confirm element-specific analytic method or source-training independence."
    }
    (ROOT/"georoc_extended_evidence_summary.json").write_text(json.dumps(overview,indent=2),encoding="utf8")
    print(json.dumps(overview,indent=2),flush=True)
    top=merged.loc[merged.candidate_gate.eq("REVIEW_METHOD_AND_SURFACE") &
        merged.country.eq("CHILE")].copy()
    top=top.drop_duplicates("citation_number").sort_values("nearest_training_km",ascending=False)
    top[["sample_name","citation_number","citation_doi","latitude","longitude","li_ppm","nearest_training_km"]].head(25).to_csv(
        ROOT/"method_review_shortlist.csv",index=False)
    print("SHORTLIST",len(top),flush=True)
if __name__=="__main__":run()

