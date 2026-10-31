"""Publication-first evidence registry for GEOROC: 59 citations, sample gates, auditable decisions."""
from pathlib import Path
import json,hashlib,re
import pandas as pd
BASE=Path("results/scientific_simulations/external_geochemistry_audit")
def main():
 refs=pd.read_csv(BASE/"all_reviewable_citations.csv")
 samples=pd.read_csv(BASE/"full_mamani_contamination_audit.csv",low_memory=False)
 selection=samples.loc[samples.candidate_gate.eq("REVIEW_METHOD_AND_SURFACE")].copy()
 selection["reviewable_after_full_source_buffer"]=selection.nearest_full_mamani_km.ge(100)
 selection["ref_id"]=pd.to_numeric(selection.citation_number,errors="coerce")
 grouped=selection.groupby("ref_id").agg(
    total=("sample_name","size"),
    away_full_mamani=("reviewable_after_full_source_buffer","sum"),
    chile=("country",lambda s:int(s.eq("CHILE").sum())),
    min_distance_full_km=("nearest_full_mamani_km","min"),
    median_li=("li_ppm","median"),
 ).reset_index()
 data=refs.merge(grouped,left_on="citation_number",right_on="ref_id",how="left",validate="one_to_one")
 for col in ("total","away_full_mamani","chile"):data[col]=data[col].fillna(0).astype(int)
 data["method_lithium_verified"]=False
 data["original_tables_checked"]=False
 data["spatial_surface_match_verified"]=False
 data["training_source_independence_proven"]=False
 data["readiness"]="NEEDS_PRIMARY_METHOD_AND_SURFACE_EVIDENCE"
 data["review_score"]=data.away_full_mamani*3+data.chile*2
 data=data.sort_values(["review_score","away_full_mamani"],ascending=False)
 data.to_csv(BASE/"publication_first_review_register.csv",index=False)
 target=data.head(10).copy()
 target.to_csv(BASE/"publication_priority_top10.csv",index=False)
 details={
   "total_references":len(data),"papers_top10":len(target),
   "samples_in_top10":int(target.total.sum()),
   "full_corpus_distant_samples_in_top10":int(target.away_full_mamani.sum()),
   "primary_methods_verified":0,"external_samples_approved":0,
   "rank_basis":"Full-Mamani09 separation >=100km and Chile coverage; publication metadata already downloaded.",
   "requirements":["Original primary article or supplements with Li analytical method for individual specimens",
       "Field/specimen provenance and observed outcrop for satellite surface validation",
       "Literature overlap check with original Mamani09 and bootstrap mirror",
       "Fixed M2 classifier label definition and held-out independent external scores"]
 }
 (BASE/"publication_first_review_manifest.json").write_text(json.dumps(details,indent=2),encoding="utf8")
 print("PUBLICATION_REVIEW",json.dumps(details),flush=True)
 print(target[["citation_number","samples","total","away_full_mamani","chile","DOI","TITLE"]].to_string(index=False,max_colwidth=80),flush=True)
if __name__=="__main__":main()
