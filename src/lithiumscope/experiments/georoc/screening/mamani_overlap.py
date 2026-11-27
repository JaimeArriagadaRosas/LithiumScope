"""Audit GEOROC vs the *full* Mamani09 raw corpus, not only M2 training subset.

Distances and chemical matches are diagnostic; require primary source methods and
surface correspondence before any sample can become labelled external validation.
"""
from __future__ import annotations
import hashlib,json
from pathlib import Path
from lithiumscope.experiments.georoc.protocol import EXPERIMENT_ROOT
from lithiumscope.core.paths import PROJECT_ROOT
from lithiumscope.datasets.validation.spatial.coordinates import assess_coordinate_aliases
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

BASE=EXPERIMENT_ROOT
MAMANI=(PROJECT_ROOT/"data/raw/model_1/Mamani09_Table_DR2.csv")
TRAIN=(PROJECT_ROOT/"data/processed/model_2/training_manifest.csv")
EARTH_KM=6371.0088


def near_full_mamani(candidates: pd.DataFrame, mamani: pd.DataFrame) -> pd.DataFrame:
    src,_=assess_coordinate_aliases(mamani)
    src["_lat"]=pd.to_numeric(src["Latitude"],errors="coerce")
    src["_lon"]=pd.to_numeric(src["Longitude"],errors="coerce")
    src["_li"]=pd.to_numeric(src["Li_icpms"],errors="coerce")
    src=src.loc[src._lat.between(-90,90)&src._lon.between(-180,180)]
    if len(src) <= 700:
        raise ValueError("Full Mamani reference corpus required")
    tree=BallTree(np.deg2rad(src[["_lat","_lon"]].to_numpy()),metric="haversine")
    out=candidates.reset_index(drop=True).copy()
    coords=out[["latitude","longitude"]].apply(pd.to_numeric,errors="coerce")
    valid=coords.latitude.between(-90,90)&coords.longitude.between(-180,180)
    if not valid.all():
        raise ValueError("Invalid candidate coordinates; review ledger before overlap audit")
    if out.empty:
        dd=np.empty((0,1));idx=np.empty((0,1),dtype=int)
    else:
        dd,idx=tree.query(np.deg2rad(coords.to_numpy()),k=1)
    closest=src.iloc[idx[:,0]].reset_index(drop=True)
    out=candidates.reset_index(drop=True).copy()
    out["nearest_full_mamani_km"]=np.round(dd[:,0]*EARTH_KM,6)
    out["nearest_full_mamani_sample_id"]=closest["Sample"].astype(str)
    out["nearest_full_mamani_li_ppm"]=closest["_li"].to_numpy()
    out["nearest_full_mamani_reference"]=closest["Geochemistry Reference"].astype(str)
    out["within_250m_full_mamani"]=out.nearest_full_mamani_km.le(.25)
    out["within_50km_full_mamani"]=out.nearest_full_mamani_km.le(50)
    out["within_100km_full_mamani"]=out.nearest_full_mamani_km.le(100)
    out["same_li_value_as_nearest"]=np.isclose(
        out["li_ppm"],out["nearest_full_mamani_li_ppm"],rtol=.0001,atol=.01,equal_nan=False)
    out["possible_same_coordinate_li"]=out.within_250m_full_mamani & out.same_li_value_as_nearest
    # Formal evidence approval is intentionally never inferred from distance or year.
    out["independent_ground_truth_approved"]=False
    out["full_corpus_gate"]=np.select(
        [out.possible_same_coordinate_li,out.within_250m_full_mamani,
         out.within_100km_full_mamani],
        ["LIKELY_SHARED_SAMPLE","COLLOCATED_MAMANI","NEAR_FULL_SOURCE"],
        default="AWAY_FROM_FULL_SOURCE_REQUIRES_ORIGINAL_PAPER")
    return out


def main(base=BASE, *, candidate_path=None, mamani_path=MAMANI, training_path=TRAIN):
    base=Path(base)
    candidate_path=Path(candidate_path) if candidate_path is not None else base/"georoc_extended_evidence_screen.csv"
    candidate=pd.read_csv(candidate_path,low_memory=False)
    mamani=pd.read_csv(mamani_path,encoding="latin1",low_memory=False)
    audited=near_full_mamani(candidate,mamani)
    path=base/"full_mamani_contamination_audit.csv"
    audited.to_csv(path,index=False)
    report={
        "source_mamani_rows":len(mamani),"candidate_rows":len(audited),
        "full_source_250m":int(audited.within_250m_full_mamani.sum()),
        "same_coordinate_and_li":int(audited.possible_same_coordinate_li.sum()),
        "full_source_50km":int(audited.within_50km_full_mamani.sum()),
        "full_source_100km":int(audited.within_100km_full_mamani.sum()),
        "post_automatic_gate_review_rows":int(audited.candidate_gate.eq("REVIEW_METHOD_AND_SURFACE").sum()),
        "post_gate_far_from_full_mamani":int((audited.candidate_gate.eq("REVIEW_METHOD_AND_SURFACE") &
                                               ~audited.within_100km_full_mamani).sum()),
        "full_source_sha256":hashlib.sha256(Path(mamani_path).read_bytes()).hexdigest(),
        "training_subset_sha256":hashlib.sha256(Path(training_path).read_bytes()).hexdigest(),
        "data_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
        "caution":"2009 original data have ambiguous coordinate precision and references. Distance only flags risk; formal validation requires geological surface sampling and original Li methods.",
        "approved_external_labels":0,
    }
    (base/"full_mamani_contamination_audit.json").write_text(
        json.dumps(report,ensure_ascii=False,indent=2),encoding="utf8")
    print("MAMANI_AUDIT",json.dumps(report,ensure_ascii=False),flush=True)
    print("GATE_COUNTS",audited.full_corpus_gate.value_counts().to_dict(),flush=True)

if __name__=="__main__":main()
