"""Offline v5 extraction and support audit on frozen M2; no fit or activation."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.paths import PROJECT_ROOT
from lithiumscope.model_2.preparation.features.physical_v5 import (
    PhysicalPolicy, extract_physical_v5, TOTAL_FEATURES,
)
from lithiumscope.model_2.preparation.features.physical_inputs import load_physical_sample

FROZEN_SHA="5c42e8b5fac697d6c1a579ef760ad642d736da14fa4e7d605596b7f4813498e1"
ROOT=PROJECT_ROOT
V4=ROOT/"results/scientific_simulations/m2_rededge_v4_20261113"
INVENTORY=ROOT/"results/scientific_simulations/sentinel_methodology_20261112/inventory"


def run(output, *, root=ROOT, policy=PhysicalPolicy()):
    output=Path(output)
    if output.exists():
        raise FileExistsError("Use a new audit directory; do not replace evidence")
    manifest_path=root/"data/processed/model_2/training_manifest.csv"
    if file_sha256(manifest_path)!=FROZEN_SHA:
        raise ValueError("Frozen M2 manifest changed")
    paths={"manifest":manifest_path,
           "qa":root/INVENTORY.relative_to(ROOT)/"local_patch_inventory_783.csv",
           "rededge_qa":root/V4.relative_to(ROOT)/"cohort_706/rededge_quality.csv",
           "v4_matrix":root/V4.relative_to(ROOT)/"cohort_706/training_v4_170.csv",
           "v4_state":root/V4.relative_to(ROOT)/"cohort_706/build_status.json"}
    manifest=pd.read_csv(paths["manifest"])
    inventory=pd.read_csv(paths["qa"]).iloc[:len(manifest)].copy()
    redqa=pd.read_csv(paths["rededge_qa"])
    v4=pd.read_csv(paths["v4_matrix"])
    state=json.loads(paths["v4_state"].read_text(encoding="utf-8"))
    if file_sha256(paths["v4_matrix"])!=state["matrix_sha256"] or state["manifest_sha256"]!=FROZEN_SHA:
        raise ValueError("Frozen v4 source changed")
    if len(manifest)!=706 or manifest.sample_id.duplicated().any():
        raise ValueError("Unexpected frozen manifest identity")
    for frame in (inventory,redqa):
        if frame.sample_id.astype(str).tolist()!=manifest.sample_id.astype(str).tolist():
            raise ValueError("QA rows do not match frozen manifest")
    if v4.shape!=(706,172) or not np.array_equal(v4.Li_icpms,manifest.Li_icpms) or not np.array_equal(v4.spatial_group,manifest.spatial_group):
        raise ValueError("Frozen v4 rows differ")
    duplicate_pairs=[(c,c.replace("nbr_","nir_swir22_nd_",1)) for c in v4 if c.startswith("nbr_")]
    if len(duplicate_pairs)!=6 or any(not np.array_equal(v4[a],v4[b]) for a,b in duplicate_pairs):
        raise ValueError("Expected six identical historical NBR descriptors")
    output.mkdir(parents=True,exist_ok=False)
    features=[]
    records=[]
    support=[]
    profiles=[]
    for i,(sample,qa,red) in enumerate(zip(manifest.itertuples(index=False),
                                          inventory.itertuples(index=False),
                                          redqa.itertuples(index=False),strict=True)):
        row={"manifest_index":i,"sample_id":str(sample.sample_id),"scene_id":str(sample.sentinel_scene_id),
             "spatial_group":str(sample.spatial_group),"Li_icpms":float(sample.Li_icpms)}
        if str(qa.image_sha256)!=str(red.original_patch_sha256) or str(qa.scene_id)!=row["scene_id"] or str(red.scene_id)!=row["scene_id"]:
            raise ValueError(f"QA source evidence differs for {sample.sample_id}")
        try:
            cube,identity=load_physical_sample(
                sample,base_sha=str(qa.image_sha256),rededge_sha=str(red.rededge_patch_sha256),
                scene_hash=str(red.scene_band_hash),scene_cache=root/INVENTORY.relative_to(ROOT)/"stac_item_cache",
                rededge_cache=root/V4.relative_to(ROOT)/"patches")
            values,quality=extract_physical_v5(cube,policy=policy)
            for name,fraction in quality.pop("descriptor_support").items():
                support.append({"sample_id":row["sample_id"],"descriptor":name,"valid_fraction":fraction})
            for name,fraction in quality.pop("band_finite_fraction").items():
                profiles.append({"sample_id":row["sample_id"],"band":name,"finite_fraction":fraction})
            records.append({**row,**identity,**quality,"reasons":";".join(quality["reasons"]),"state":"extracted"})
            features.append({"sample_id":row["sample_id"],**values})
        except (OSError,ValueError,KeyError,TypeError) as exc:
            records.append({**row,"state":"blocked_input","eligible_for_experimental_fit":False,"reasons":str(exc)})
            features.append({"sample_id":row["sample_id"]})
        if (i+1)%100==0:
            print(f"Physical extraction {i+1}/706",flush=True)
    audit=pd.DataFrame(records)
    matrix=pd.DataFrame(features)
    if matrix.shape!=(706,TOTAL_FEATURES+1) or matrix.sample_id.tolist()!=manifest.sample_id.astype(str).tolist():
        raise ValueError("Physical feature schema or cohort identity incomplete")
    expected=[c for c in v4 if c not in {"Li_icpms","spatial_group"} and not c.startswith("nir_swir22_nd_")]
    if list(matrix.columns[1:])!=expected:
        raise ValueError("Physical schema differs from planned 170-6 descriptor order")
    scores=pd.DataFrame(support)
    bands=pd.DataFrame(profiles)
    if scores.empty or bands.empty:
        raise ValueError("No verified feature evidence")
    audit.to_csv(output/"sample_audit.csv",index=False)
    matrix.to_csv(output/"features_v5_164.csv",index=False)
    scores.to_csv(output/"descriptor_support.csv",index=False)
    bands.to_csv(output/"band_support.csv",index=False)
    by_scene=audit.groupby("scene_id",dropna=False).agg(
        samples=("sample_id","size"),eligible=("eligible_for_experimental_fit","sum"))
    by_scene.to_csv(output/"scene_summary.csv")
    scores.groupby("descriptor").valid_fraction.agg(["count","min","median","mean"]).to_csv(output/"support_summary.csv")
    old=v4.loc[:,expected]
    finite=np.isfinite(matrix[expected].to_numpy(float)).all(axis=1)
    result={"scope":"Retrospective physical representation only; not a trained model, holdout or approved cohort",
            "policy":asdict(policy),"source_sha256":{k:file_sha256(v) for k,v in paths.items()},
            "samples":len(audit),"verified_extracted":int(audit.state.eq("extracted").sum()),
            "blocked_input":int(audit.state.ne("extracted").sum()),"feature_count":TOTAL_FEATURES,
            "rows_with_all_finite_descriptors":int(finite.sum()),
            "experiment_fit_eligible":int(audit.eligible_for_experimental_fit.sum()),
            "retained_all_rows":True,"legacy_v4_unchanged":file_sha256(paths["v4_matrix"])==state["matrix_sha256"],
            "removed_duplicate_columns":[b for _,b in duplicate_pairs],
            "raw_finite_zero_values":int(audit.raw_finite_zero_pixels.fillna(0).sum()),
            "physical_negative_values":int(audit.physical_negative_values.fillna(0).sum()),
            "physical_zero_values":int(audit.physical_zero_values.fillna(0).sum()),
            "v4_vs_v5_changed_descriptor_count":int(np.any(~np.isclose(old.to_numpy(float)[finite],
                matrix[expected].to_numpy(float)[finite],rtol=1e-7,atol=1e-8),axis=0).sum()),
            "scl_applied":False,"field_rock_exposure_certified":False,
            "artifacts":{p.name:file_sha256(p) for p in sorted(output.glob("*.csv"))}}
    (output/"summary.json").write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args()
    print(json.dumps(run(args.output),indent=2,ensure_ascii=False))
