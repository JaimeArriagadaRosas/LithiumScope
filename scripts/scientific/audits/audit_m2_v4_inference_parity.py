"""Offline M2 v4 inference parity against the exact historical matrix, never fit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import joblib
import numpy as np
import pandas as pd

from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.paths import PROJECT_ROOT
from lithiumscope.datasets.validation.paths import resolve_image_path
from lithiumscope.model_2.prediction.frozen_rededge import CATALOG, MANIFEST_SHA, QUALITY_SHA, V4_DIR
from lithiumscope.model_2.prediction.spectral_contract import prepare_spectral_input, SpectralAbstention
from lithiumscope.model_2.prediction.predictor import score_model_2_image

MODEL_SHA="794b0c447f8e00e608ac1457546943a804dd46cc2bf1b7c573887651514d4770"
MATRIX_SHA="f4d1f5809fb1242f27d0bbc2348e9dab2e77a017ea77049090652929244a9513"


def run(model,output,*,trust_local_bundle=False):
    model,output=Path(model).resolve(),Path(output).resolve()
    root=PROJECT_ROOT.resolve()
    if not trust_local_bundle:
        raise ValueError("Only explicitly trusted local bundle may be loaded")
    if not output.is_relative_to(root) or model.is_relative_to(output):
        raise ValueError("Use isolated diagnostic directory in current clone")
    if file_sha256(model)!=MODEL_SHA:
        raise ValueError("Frozen M2 v4 bundle differs")
    meta_path=model.with_name("metadata.json")
    metadata=json.loads(meta_path.read_text(encoding="utf-8"))
    if metadata["model_sha256"]!=MODEL_SHA or not metadata["experimental_only"]:
        raise ValueError("M2 v4 metadata does not identify the experimental bundle")
    files={
        root/CATALOG/"training_manifest.csv":MANIFEST_SHA,
        root/CATALOG/"rededge_quality.csv":QUALITY_SHA,
        root/V4_DIR/"cohort_706/training_v4_170.csv":MATRIX_SHA,
        model:MODEL_SHA,meta_path:file_sha256(meta_path),
    }
    if any(file_sha256(path)!=sha for path,sha in files.items()):
        raise ValueError("Frozen inference evidence drift")
    m=pd.read_csv(root/CATALOG/"training_manifest.csv")
    q=pd.read_csv(root/CATALOG/"rededge_quality.csv")
    x=pd.read_csv(root/V4_DIR/"cohort_706/training_v4_170.csv")
    inv_path=root/"results/scientific_simulations/sentinel_methodology_20261112/inventory/local_patch_inventory_783.csv"
    files[inv_path]=file_sha256(inv_path)
    inv=pd.read_csv(inv_path).iloc[:706]
    if not (len(m)==len(q)==len(x)==len(inv)==706 and m.sample_id.equals(q.sample_id)
            and m.sample_id.equals(inv.sample_id) and q.original_patch_sha256.equals(inv.image_sha256)
            and m.sentinel_scene_id.equals(q.scene_id)
            and np.array_equal(m.Li_icpms,x.Li_icpms)
            and m.spatial_group.equals(x.spatial_group)):
        raise ValueError("Frozen manifest/matrix/QA alignment differs")
    output.mkdir(parents=True,exist_ok=False)
    local_model=output/"model.joblib"
    shutil.copy2(model,local_model)
    if file_sha256(local_model)!=MODEL_SHA:
        raise ValueError("Bundle copy differs")
    bundle=joblib.load(local_model)
    features=list(bundle["features"])
    if bundle.get("feature_extractor_version")!=4 or not bundle.get("experimental_only") or len(features)!=170:
        raise ValueError("Expected historical experimental 170-feature M2 model")
    if not np.array_equal(bundle["estimator"].classes_,[0,1]):
        raise ValueError("Class order differs")
    cutoff=float(bundle["quality_min_joint_valid"])
    eligible=(inv.sixband_joint_valid_fraction.to_numpy()>=cutoff)&(q.rededge_joint_valid_fraction.to_numpy()>=cutoff)
    reference=bundle["estimator"].predict_proba(x[features])[:,1]
    rows=[]
    reconstructed=[]
    positions=[]
    for i,sample in enumerate(m.itertuples(index=False)):
        patch=resolve_image_path(sample.image_path)
        if file_sha256(patch)!=q.original_patch_sha256.iloc[i]:
            raise ValueError(f"Base patch drift: {sample.sample_id}")
        row={"row_position":i,"sample_id":sample.sample_id,"scene_id":sample.sentinel_scene_id,
             "expected_qa_pass":bool(eligible[i]),"reference_matrix_score":float(reference[i])}
        try:
            frame,diag=prepare_spectral_input(patch,bundle,allow_acquisition=False)
        except SpectralAbstention as exc:
            if eligible[i] or exc.status not in {"ABSTAIN_LOW_SPECTRAL_COVERAGE","ABSTAIN_LOW_REDEDGE_COVERAGE"}:
                raise ValueError(f"Unexpected abstention: {sample.sample_id}: {exc.status}: {exc}") from exc
            row.update(status=exc.status,reason=str(exc))
        else:
            if not eligible[i] or diag.get("rededge_cache_source")!="verified_frozen_v4_706":
                raise ValueError(f"Unexpected score or source: {sample.sample_id}")
            actual,expected=frame.to_numpy(float),x.loc[[i],features].to_numpy(float)
            if not np.allclose(actual,expected,rtol=1e-12,atol=5e-13):
                raise ValueError(f"170-feature reconstruction drift: {sample.sample_id}")
            row.update(status="SCORED",maximum_feature_absolute_delta=float(np.max(np.abs(actual-expected))),
                       rededge_cache_source=diag["rededge_cache_source"],
                       rededge_patch_sha256=diag["rededge_patch_sha256"],input_contract_status=diag["input_contract_status"])
            reconstructed.append(frame)
            positions.append(i)
        rows.append(row)
        if (i+1)%100==0:
            print(f"V4_PARITY {i+1}/706",flush=True)
    values=pd.concat(reconstructed,ignore_index=True)
    scored=bundle["estimator"].predict_proba(values)[:,1]
    delta=np.abs(scored-reference[positions])
    if not np.allclose(scored,reference[positions],rtol=0,atol=1e-12):
        raise ValueError("V4 reconstructed matrix scores differ")
    for i,s,d in zip(positions,scored,delta,strict=True):
        rows[i].update(reconstructed_score=float(s),absolute_score_delta=float(d))
    # Verify the public scorer separately on ten known successful inputs and all abstentions.
    selected=list(np.asarray(positions)[np.linspace(0,len(positions)-1,10,dtype=int)])+list(np.flatnonzero(~eligible))
    for i in selected:
        patch=resolve_image_path(m.image_path.iloc[i])
        payload,diag=score_model_2_image(patch,bundle=bundle,allow_acquisition=False)
        if eligible[i]:
            if not np.isclose(payload["prospectivity_score"],reference[i],rtol=0,atol=1e-12):
                raise ValueError("Public scorer parity differs")
        elif payload["prospectivity_score"] is not None or payload["quality_status"]!=rows[i]["status"]:
            raise ValueError("Public scorer did not abstain")
        rows[i]["public_scorer_verified"]=True
    if any(file_sha256(path)!=sha for path,sha in files.items()) or file_sha256(local_model)!=MODEL_SHA:
        raise ValueError("Evidence changed during inference audit")
    table=pd.DataFrame(rows)
    table.to_csv(output/"cohort_inference_comparison.csv",index=False)
    summary={"scope":"Known training cohort compatibility only, not external validation or OOF evaluation",
             "model_sha256":MODEL_SHA,"matrix_sha256":MATRIX_SHA,"samples":len(m),
             "scores":len(positions),"abstentions":int((~eligible).sum()),
             "abstention_statuses":table.loc[~eligible,"status"].value_counts().to_dict(),
             "features":len(features),"maximum_feature_absolute_delta":float(table.maximum_feature_absolute_delta.max()),
             "maximum_score_absolute_delta":float(delta.max()),"public_scorer_checks":len(selected),
             "geometry_certification":"legacy_unverified_geometry","experimental_only":True,
             "network_acquisition":False,"fit":False,"activation":False,"source_files_verified_unchanged":True}
    (output/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--trust-local-bundle",action="store_true")
    args=parser.parse_args()
    run(args.model,args.output,trust_local_bundle=args.trust_local_bundle)
