"""Conservative reference/location audit before any external validation."""
from pathlib import Path
from lithiumscope.experiments.georoc.protocol import EXPERIMENT_ROOT
import hashlib,json,re
import pandas as pd
import numpy as np

ROOT=EXPERIMENT_ROOT


def main(root=ROOT, *, candidate_path=None):
    root=Path(root)
    candidate_path=Path(candidate_path) if candidate_path is not None else root/"georoc_andean_candidates.csv"
    df=pd.read_csv(candidate_path)
    # Preserve missing reference and invalid coordinate cases as review evidence.
    df["nearest_training_km"]=pd.to_numeric(df.nearest_training_km,errors="coerce")
    df["latitude"]=pd.to_numeric(df.latitude,errors="coerce")
    df["longitude"]=pd.to_numeric(df.longitude,errors="coerce")
    valid=df.latitude.between(-90,90)&df.longitude.between(-180,180)&np.isfinite(df.nearest_training_km)&df.nearest_training_km.ge(0)
    df["location_valid_for_screen"]=valid
    # Original input path is explicit.

    df["country"]=df.location.astype("string").str.extract(r"/ (CHILE|ARGENTINA|BOLIVIA|PERU|ECUADOR|COLOMBIA)(?: /|$)",expand=False)
    df["citation_key"]=df.citation_ids.fillna("").astype(str).map(lambda x:";".join(sorted(set(re.findall(r"\d+",x)))))
    df["sample_key"]=df.sample_name.astype(str).str.lower().str.replace(r"\s*\[\d+\]","",regex=True).str.strip()
    df["exact_duplicate_key"]=df.sample_key+"|"+df.latitude.round(4).astype(str)+"|"+df.longitude.round(4).astype(str)
    dup=df.duplicated("exact_duplicate_key",keep=False)
    df["duplicate_identifier_location"]=dup
    df["spatial_overlap_50km"]=df.nearest_training_km.lt(50)
    df["spatial_buffer_100km"]=df.nearest_training_km.ge(100)
    df["geographic_screen"]=np.select([~valid,df.nearest_training_km.lt(50),df.nearest_training_km.lt(100)],["INVALID_COORDINATE_OR_DISTANCE","EXCLUDE_NEAR_TRAINING","HOLD_BUFFER_50_100KM"],default="DISTANT_CANDIDATE")
    df["method_verified"]=False
    df["bibliographic_independence_verified"]=False
    df["external_holdout_status"]="CANDIDATE_NOT_APPROVED"
    df["review_priority"]=np.where((df.country=="CHILE")&df.spatial_buffer_100km&(~dup)&df.citation_key.ne(""),"HIGH_PRIORITY_REVIEW","DEFER")
    df.to_csv(root/"georoc_provenance_screen.csv",index=False)
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
    pd.DataFrame(chosen, columns=df.columns).to_csv(root/"pilot_priority_candidates.csv",index=False)
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
     "source_sha256":hashlib.sha256(candidate_path.read_bytes()).hexdigest()}
    (root/"provenance_screen_summary.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf8")
    print("SUMMARY",payload,flush=True)
    print("PILOT",pd.DataFrame(chosen, columns=df.columns)[["sample_name","citation_ids","latitude","longitude","li_ppm","nearest_training_km"]].to_string(index=False),flush=True)

if __name__=="__main__":
    from lithiumscope.core.logging.session import run_log
    with run_log("simulations","georoc_provenance_screen"):
        main()
