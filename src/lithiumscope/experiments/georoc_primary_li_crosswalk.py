"""Exact supplementary-table Li crosswalk for GEOROC citation 20576."""
from pathlib import Path
import re,json,hashlib
import numpy as np,pandas as pd
B=Path("results/scientific_simulations/external_geochemistry_audit")
s=pd.read_csv(B/"georoc_extended_evidence_screen.csv",low_memory=False)
src=s.loc[s.citation_number.eq(20576)&s.candidate_gate.eq("REVIEW_METHOD_AND_SURFACE")].copy()
path=B/"publication_sources/20576_supplement.xlsx"
def clean(v):
 v=re.sub(r"\s*\[\d+\]","",str(v),flags=re.I).replace("s_","")
 return re.sub(r"[^a-z0-9]","",v.lower())
records=[]
for sheet in ("Data for new samples","Samples with new ICPMS data"):
 df=pd.read_excel(path,sheet_name=sheet)
 df["normalized"]=df["Sample Name"].map(clean)
 df["li_num"]=pd.to_numeric(df["Li"],errors="coerce")
 df["passes_filter"]=pd.to_numeric(df["Passes Filters? (1=yes, 0=no)"],errors="coerce")
 for _,r in df.iterrows():
  records.append({"sheet":sheet,"name":r["normalized"],"li_original":r["li_num"],"passes_filter":r["passes_filter"],
   "lat_original":pd.to_numeric(r.get("Lat"),errors="coerce"),"lon_original":pd.to_numeric(r.get("Long"),errors="coerce"),
   "sample_name_supp":r["Sample Name"]})
tab=pd.DataFrame(records)
src["name"]=src.sample_name.map(clean)
joined=src.merge(tab,on="name",how="left")
joined["same_li"]=np.isclose(joined.li_ppm,joined.li_original,rtol=.00001,atol=.02,equal_nan=False)
joined["coords_match"]=np.isclose(joined.latitude,joined.lat_original,atol=.01,equal_nan=False) & np.isclose(joined.longitude,joined.lon_original,atol=.01,equal_nan=False)
joined["source_value_status"]=np.where(joined.same_li,"PRIMARY_LI_VALUE_MATCH","NO_EXACT_PRIMARY_LI_MATCH")
joined["scientific_approval"]=False
joined[["sample_name","li_ppm","sample_name_supp","li_original","same_li","coords_match","sheet","passes_filter","source_value_status"]].to_csv(B/"georoc_20576_primary_li_crosswalk.csv",index=False)
report={"georoc_candidates":len(src),"matched_rows":int(joined.same_li.sum()),"unique_primary_matches":int(joined.loc[joined.same_li,"sample_name"].nunique()),"coordinate_matches":int((joined.same_li&joined.coords_match).sum()),"primary_spreadsheet_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"approved_for_external_model2":0}
(B/"georoc_20576_primary_li_crosswalk.json").write_text(json.dumps(report,indent=2),encoding="utf8")
print("CROSSWALK",report,flush=True)
print(joined.loc[joined.same_li,["sample_name","li_ppm","li_original","lat_original","lon_original","latitude","longitude","passes_filter"]].head(8).to_string(index=False),flush=True)


