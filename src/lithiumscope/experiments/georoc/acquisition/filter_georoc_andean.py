"""Historical GEOROC candidate screening; distance alone is not independence."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
from lithiumscope.core.paths import PROJECT_ROOT,RESULTS_DIR
from lithiumscope.core.hashing import file_sha256
from lithiumscope.datasets.adapters.georoc_dataverse import whole_rock_candidates
from lithiumscope.results.atomic_exports import write_json,write_table


def run(root,training_manifest):
    root=Path(root)
    paths=sorted((root/"georoc_andean_raw").glob("*.csv"))
    if not paths:
        raise FileNotFoundError("No local GEOROC partitions")
    training=pd.read_csv(training_manifest)
    coordinates=training[["latitude","longitude"]].to_numpy(float)
    if not len(coordinates) or not np.isfinite(coordinates).all() or (np.abs(coordinates)>[90,180]).any():
        raise ValueError("Invalid training coordinates")
    tree=BallTree(np.deg2rad(coordinates),metric="haversine")
    rows,stats=[],[]
    for path in paths:
        candidates,audit=whole_rock_candidates(pd.read_csv(path,low_memory=False,encoding="latin1"),path.name)
        if len(candidates):
            distance,_=tree.query(np.deg2rad(candidates[["latitude","longitude"]].to_numpy()),k=1)
            candidates["nearest_training_km"]=np.round(distance[:,0]*6371,3)
        else:
            candidates["nearest_training_km"]=pd.Series(dtype=float)
        rows.append(candidates)
        stats.append({"file":path.name,**audit,"sha256":file_sha256(path)})
    all_=pd.concat(rows,ignore_index=True).drop_duplicates(subset=["sample_name","latitude","longitude","li_ppm"])
    all_["possible_overlap_50km"]=all_.nearest_training_km.le(50)
    all_["audit_status"]="CANDIDATE_PROVENANCE_AND_METHOD_UNVERIFIED"
    write_table(root/"georoc_andean_candidates.csv",all_)
    summary={"source":"https://doi.org/10.25625/PVFZCE","partitions":stats,
        "training_manifest_sha256":file_sha256(training_manifest),"candidate_count":len(all_),
        "within_50km":int(all_.possible_overlap_50km.sum()),"at_least_50km":int((~all_.possible_overlap_50km).sum()),
        "independent_accepted":0,"exclusion_note":"Method, reference and sample overlap audit incomplete."}
    write_json(root/"georoc_andean_audit.json",summary)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=RESULTS_DIR/"scientific_simulations/external_geochemistry_audit")
    parser.add_argument("--training-manifest",type=Path,default=PROJECT_ROOT/"data/processed/model_2/training_manifest.csv")
    args=parser.parse_args()
    print(run(args.root,args.training_manifest))


if __name__=="__main__":
    main()
