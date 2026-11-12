from pathlib import Path
import pandas as pd,numpy as np,json,hashlib
from sklearn.neighbors import BallTree
root=Path("results/scientific_simulations/external_geochemistry_audit")
paths=sorted((root/"georoc_andean_raw").glob("*.csv"))
training=pd.read_csv(r"D:\Descargas\LithiumScope\data\processed\model_2\training_manifest.csv")
tree=BallTree(np.deg2rad(training[["latitude","longitude"]].to_numpy()),metric="haversine")
rows=[];stats=[]
for path in paths:
    df=pd.read_csv(path,low_memory=False,encoding="latin1")
    lat=pd.to_numeric(df["LATITUDE MIN"],errors="coerce")
    lon=pd.to_numeric(df["LONGITUDE MIN"],errors="coerce")
    li=pd.to_numeric(df["LI(PPM)"],errors="coerce")
    material=df["MATERIAL"].fillna("").astype(str)
    mask=li.notna()&li.ge(0)&lat.between(-90,90)&lon.between(-180,180)&material.str.contains(r"\bWR\b|WHOLE ROCK",case=False,regex=True)
    selected=df.loc[mask].copy()
    distance,_=tree.query(np.deg2rad(np.column_stack([lat[mask],lon[mask]])),k=1)
    output=pd.DataFrame({"source_file":path.name,"sample_name":selected["SAMPLE NAME"].astype(str),
        "latitude":lat[mask].to_numpy(),"longitude":lon[mask].to_numpy(),"li_ppm":li[mask].to_numpy(),
        "material":selected["MATERIAL"].astype(str),"citation_ids":selected["CITATIONS"].astype(str),
        "location":selected["LOCATION"].astype(str),
        "nearest_training_km":np.round(distance[:,0]*6371,3)})
    rows.append(output)
    stats.append({"file":path.name,"all_rows":len(df),"li_numeric":int(li.notna().sum()),"candidate_whole_rock":len(output),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
all_=pd.concat(rows,ignore_index=True)
all_=all_.drop_duplicates(subset=["sample_name","latitude","longitude","li_ppm"])
all_["possible_overlap_50km"]=all_.nearest_training_km.le(50)
all_["audit_status"]="CANDIDATE_PROVENANCE_AND_METHOD_UNVERIFIED"
all_.to_csv(root/"georoc_andean_candidates.csv",index=False)
summary={"source":"https://doi.org/10.25625/PVFZCE","dataset":"GEOROC Compilation Convergent Margins 2026-09","partitions":stats,
"candidate_count":len(all_),"within_50km":int(all_.possible_overlap_50km.sum()),"at_least_50km":int((~all_.possible_overlap_50km).sum()),
"independent_accepted":0,"exclusion_note":"Method, reference and sample overlap audit incomplete; geographic distance alone does not prove independence."}
(root/"georoc_andean_audit.json").write_text(json.dumps(summary,indent=2),encoding="utf8")
print("STATS",summary,flush=True);print("EXAMPLE",all_.head(4).to_string(index=False,max_cols=8),flush=True)

