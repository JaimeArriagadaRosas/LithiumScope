"""Conservative reference/location audit before any external validation."""
from pathlib import Path
import hashlib,json,re
import pandas as pd
from sklearn.cluster import DBSCAN
import numpy as np

ROOT=Path("results/scientific_simulations/external_geochemistry_audit")
df=pd.read_csv(ROOT/"georoc_andean_candidates.csv")
df["country"]=df.location.str.extract(r"/ (CHILE|ARGENTINA|BOLIVIA|PERU|ECUADOR|COLOMBIA)(?: /|$)",expand=False)
df["citation_key"]=df.citation_ids.fillna("").map(lambda x:";".join(sorted(set(re.findall(r"\d+",x)))))
df["sample_key"]=df.sample_name.astype(str).str.lower().str.replace(r"\s*\[\d+\]","",regex=True).str.strip()
df["exact_duplicate_key"]=df.sample_key+"|"+df.latitude.round(4).astype(str)+"|"+df.longitude.round(4).astype(str)
dup=df.duplicated("exact_duplicate_key",keep=False)
df["duplicate_identifier_location"]=dup
df["spatial_overlap_50km"]=df.nearest_training_km.lt(50)
df["spatial_buffer_100km"]=df.nearest_training_km.ge(100)
df["geographic_screen"]=np.where(df.nearest_training_km.lt(50),"EXCLUDE_NEAR_TRAINING",np.where(df.nearest_training_km.lt(100),"HOLD_BUFFER_50_100KM","DISTANT_CANDIDATE"))
df["method_verified"]=False
df["bibliographic_independence_verified"]=False
df["external_holdout_status"]="CANDIDATE_NOT_APPROVED"
df["review_priority"]=np.where((df.country=="CHILE")&df.spatial_buffer_100km&(~dup)&df.citation_key.ne(""),"HIGH_PRIORITY_REVIEW","DEFER")
df.to_csv(ROOT/"georoc_provenance_screen.csv",index=False)
pilot=df.loc[df.review_priority.eq("HIGH_PRIORITY_REVIEW")].copy()
pilot=pilot.sort_values(["nearest_training_km","citation_key","sample_key"],ascending=[False,True,True])
# Pick distinct references and locations, no inferred independence.
chosen=[];refs=set();coords=[]
for rec in pilot.to_dict("records"):
    ref=rec["citation_key"]
    if ref in refs:continue
    if any(abs(rec["latitude"]-x)<.45 and abs(rec["longitude"]-y)<.45 for x,y in coords):continue
    chosen.append(rec);refs.add(ref);coords.append((rec["latitude"],rec["longitude"]))
    if len(chosen)==8:break
pd.DataFrame(chosen).to_csv(ROOT/"pilot_priority_candidates.csv",index=False)
payload={
 "raw_candidates":len(df),"country_counts":df.country.fillna("UNCLASSIFIED").value_counts().to_dict(),
 "exact_identifier_location_duplicate_rows":int(dup.sum()),
 "less_than_50km":int(df.spatial_overlap_50km.sum()),
 "between_50_and_100km":int(df.nearest_training_km.between(50,100,inclusive="left").sum()),
 "at_least_100km":int(df.spatial_buffer_100km.sum()),
 "chile_far_100km":int(((df.country=="CHILE")&df.spatial_buffer_100km).sum()),
 "high_priority_chile_candidates":int(df.review_priority.eq("HIGH_PRIORITY_REVIEW").sum()),
 "pilot_candidates":len(chosen),"external_accepted":0,
 "note":"Geographic, identifier and citation-string screening only. Full methodological and bibliographic independence not established.",
 "source_sha256":hashlib.sha256((ROOT/"georoc_andean_candidates.csv").read_bytes()).hexdigest()}
(ROOT/"provenance_screen_summary.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf8")
print("SUMMARY",payload,flush=True)
print("PILOT",pd.DataFrame(chosen)[["sample_name","citation_ids","latitude","longitude","li_ppm","nearest_training_km"]].to_string(index=False),flush=True)
