"""Bounded physically-corrected scientific M2 comparison on frozen 706 held-outs.

Uses immutable 783-row manifest, 111 local STAC items (source metadata) and
593 original patch arrays. DOES NOT use new GEOROC labels for fitting:
each CatBoost fit uses only original historical TRAIN indices of its fold.

Arms (predeclared; compared to archived CatBoost Phase C reference):
  raw_valid_only: DN original; no imputation
  reflectance_valid_only: scale/offset STAC; no imputation (scientific)
  reflectance_median_filled: scale/offset STAC; legacy median fill control

No tuning, calibration, external test claims, active model changes or commits.
This experiment is exploratory because the 706 historical OOF labels have
been used repeatedly in previous method development.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from lithiumscope.core.paths import PROJECT_ROOT
from lithiumscope.model_2.methodology.ablations import (
    PARAMS, HISTORICAL_N, build_inputs, define_outer_fold, fit_classifier,
    score_metrics, bootstrap_group_delta
)
from lithiumscope.model_2.methodology.radiometric_features import build_feature_variants
from lithiumscope.model_2.methodology.satellite_quality import scene_metadata

ROOT_CACHE="results/scientific_simulations/sentinel_methodology_20261112/inventory/stac_item_cache"
BASELINE="results/scientific_simulations/m2_phase_c_20261112/pilot_8arms/oof_legacy_all.csv"
ARMS=("legacy_archived","raw_valid_only",
      "reflectance_valid_only","reflectance_median_filled")


def build_feature_caches(
    root:Path,output:Path, *, contract:pd.DataFrame,
    frozen_columns:list[str], external_sha_map:dict|None=None,
)->tuple[dict[str,pd.DataFrame],dict]:
    cache_path=root/ROOT_CACHE
    if not cache_path.is_dir():
        raise FileNotFoundError("Source STAC metadata is missing. Run online inventory first")
    scene_cache={}
    rows={name:[] for name in ARMS[1:]}
    original_feature_cache={}
    seen_pairs={}
    coverage=[]
    for i,sample in contract.iterrows():
        scene_id=str(sample.sentinel_scene_id)
        if scene_id not in scene_cache:
            file=cache_path/(scene_id+".json")
            if not file.is_file():raise FileNotFoundError(f"STAC scene metadata missing {scene_id}")
            item=json.loads(file.read_text(encoding="utf-8"))
            if str(item.get("id"))!=scene_id:
                raise ValueError("STAC scene id mismatch")
            scene_cache[scene_id]=scene_metadata(item)
        image=Path(str(sample.image_path))
        if not image.is_file():raise FileNotFoundError(image)
        # Same metadata AND same bytes => exactly the same physical transform.
        raw_sha=hashlib.sha256(image.read_bytes()).hexdigest()
        key=(raw_sha,scene_id)
        if key not in original_feature_cache:
            cube=np.load(image,allow_pickle=False)
            if cube.shape!=(6,32,32):raise ValueError("Unexpected spectral shape")
            valid=np.isfinite(cube).all(axis=0)
            variants=build_feature_variants(cube,scene_cache[scene_id])
            for name,vec in variants.items():
                if list(vec)!=frozen_columns or len(vec)!=126:
                    raise ValueError("Changed frozen 126-feature schema")
            original_feature_cache[key]=variants
            seen_pairs[key]=float(valid.mean())
        else:
            variants=original_feature_cache[key]
        for name in rows:
            rows[name].append([variants[name][k] for k in frozen_columns])
        coverage.append({
            "manifest_index":i,"sample_id":str(sample.sample_id),
            "image_sha256":raw_sha,
            "scene_id":scene_id,
            "joint_spectral_valid_fraction":seen_pairs[key],
            "scene_radiometric_metadata_present":True,
        })
        if i%100==0:print("FEATURE_EXTRACTION",i+1,783,flush=True)
    frames={
        name:pd.DataFrame(data,columns=frozen_columns)
        for name,data in rows.items()
    }
    for name,frame in frames.items():
        if frame.shape!=(783,126) or not np.isfinite(frame.to_numpy(float)).all():
            raise ValueError(f"Feature coverage/schema invalid: {name}")
        frame.to_csv(output/f"spectral_features_{name}.csv",index=False)
    pd.DataFrame(coverage).to_csv(output/"radiometric_input_trace.csv",index=False)
    summary={
        "rows":783,
        "unique_image_scene_pairs":len(original_feature_cache),
        "scenes_with_stac_radiometry":len(scene_cache),
        "schema_features":126,
        "original_data_touched":False,
        "feature_files":list(frames),
        "stac_files_used":len(scene_cache),
        "all_expected_scene_metadata":True,
    }
    (output/"feature_derivation_audit.json").write_text(
        json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    print("ALL_FEATURE_CACHES_COMPLETE",summary,flush=True)
    return frames,summary


def run(root:Path,out:Path, *,extract_only=False):
    contract,legacy,folds,_,hashes=build_inputs(root)
    out.mkdir(parents=True,exist_ok=True)
    reference=pd.read_csv(root/BASELINE).sort_values("test_index")
    if reference.test_index.tolist()!=list(range(706)):
        raise ValueError("Baseline must contain same 706 outer test indices")
    predictions=[]
    frozen=reference.copy()
    frozen["arm"]="legacy_archived"
    predictions.extend(frozen.to_dict("records"))
    design={
        "schema":1,"params":PARAMS,"arms":list(ARMS),
        "frozen_historical_test_indices":706,"folds":5,
        "frozen_outer_test_unchanged":True,
        "historical_train_only":True,
        "new_georoc_provisional_training_labels_used":False,
        "known_demo_ten_cases_used":False,
        "scene_replacement_used":False,
        "training_source_sha256":hashes,
        "no_activation":True,"no_hyperparameter_search":True,
        "score_threshold_for_descriptive_recall":.5,
        "primary_metric_roc_auc_ap_brier":True,
        "warning":"Exploratory: 706 old samples repeatedly used for development",
    }
    (out/"experimental_design.json").write_text(
        json.dumps(design,ensure_ascii=False,indent=2),encoding="utf-8")
    all_features,feature_summary=build_feature_caches(
        root,out,contract=contract,frozen_columns=list(legacy.columns))
    if extract_only:
        return feature_summary
    metrics_by_fold=[]
    for arm,frame in all_features.items():
        arm_rows=[]
        for fold_number,fold in enumerate(folds,1):
            tr_all,tr_hist,te,threshold=define_outer_fold(fold,contract)
            y_train=(contract.Li_icpms.iloc[tr_hist].to_numpy(float)>=threshold).astype(int)
            y_test=(contract.Li_icpms.iloc[te].to_numpy(float)>=threshold).astype(int)
            probabilities=fit_classifier(
                frame.iloc[tr_hist],y_train,frame.iloc[te])
            score=score_metrics(y_test,probabilities)
            metrics_by_fold.append({
                "arm":arm,"fold":fold_number,
                "train_rows":len(tr_hist),"test_rows":len(te),**score,
            })
            for ix,y,prob in zip(te,y_test,probabilities):
                arm_rows.append({
                    "arm":arm,"fold":fold_number,"test_index":int(ix),
                    "sample_id":str(contract.sample_id.iloc[ix]),
                    "spatial_group":str(contract.spatial_group.iloc[ix]),
                    "site_screening_id":str(contract.site_screening_id.iloc[ix]),
                    "latitude":float(contract.latitude.iloc[ix]),
                    "li_ppm":float(contract.Li_icpms.iloc[ix]),
                    "fold_li_threshold_ppm":threshold,
                    "y_true":int(y),"score":float(prob),
                    "n_train":len(tr_hist),"n_features":126,
                })
            print("FOLD",arm,fold_number,"AUC",score["auc"],
                  "AP",score["ap"],flush=True)
        pred=pd.DataFrame(arm_rows).sort_values("test_index")
        if pred.test_index.tolist()!=list(range(706)):
            raise ValueError("Outer hold-out coverage changed")
        if not np.array_equal(pred.y_true.to_numpy(int),
                              reference.y_true.to_numpy(int)):
            raise ValueError("Outer fold historical labels changed")
        if not np.array_equal(pred.fold.to_numpy(int),
                              reference.fold.to_numpy(int)):
            raise ValueError("Outer fold assignment changed")
        pred.to_csv(out/f"oof_{arm}.csv",index=False)
        predictions.extend(pred.to_dict("records"))
        (out/"progress.json").write_text(json.dumps({
            "state":"running","completed_arm":arm
        },indent=2),encoding="utf-8")
    pd.DataFrame(metrics_by_fold).to_csv(out/"fold_metrics.csv",index=False)
    paired=pd.DataFrame(predictions)
    paired.to_csv(out/"all_arms_paired_oof.csv",index=False)
    summary=[]
    for name,rows in paired.groupby("arm",sort=False):
        item={"arm":name,**score_metrics(rows.y_true,rows.score)}
        summary.append(item)
    pd.DataFrame(summary).to_csv(out/"paired_comparison.csv",index=False)
    boots=[]
    for arm in ARMS[1:]:
        boots.append(bootstrap_group_delta(
            paired,ARMS[0],arm,repeats=320,seed=20261112))
    pd.DataFrame(boots).to_csv(out/"bootstrap_group_comparisons.csv",index=False)
    # Feature shifts are descriptive, no column selection or threshold fitting.
    baseline_spectral=legacy.to_numpy(float)
    delta_rows=[]
    for name,features in all_features.items():
        numbers=features.to_numpy(float)
        diff=np.abs(numbers-baseline_spectral)
        denom=np.maximum(np.median(np.abs(baseline_spectral),axis=0),1e-6)
        delta_rows.append({
            "arm":name,
            "mean_absolute_feature_delta_vs_legacy":float(diff.mean()),
            "median_relative_delta_across_features":float(
                np.median(np.median(diff,axis=0)/denom)),
            "all_features_finite":bool(np.isfinite(numbers).all()),
        })
    pd.DataFrame(delta_rows).to_csv(out/"spectral_feature_shift.csv",index=False)
    result={
        "state":"completed","rows":706,"positive":int(reference.y_true.sum()),
        "fit_count":3*len(folds),"all_arms":list(ARMS),
        "original_patches_changed":False,"scene_replacements":0,
        "active_model_changed":False,"external_holdout":False,
        "new_georoc_used_for_fit":False,
        "old_ten_demo_used":False,
    }
    (out/"experiment_summary.json").write_text(
        json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
    print("RADIOMETRIC_EXPERIMENT_COMPLETE",result,flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",required=True)
    p.add_argument("--extract-only",action="store_true")
    args=p.parse_args()
    out=Path(args.output)
    if not out.is_absolute():out=PROJECT_ROOT/out
    run(PROJECT_ROOT,out,extract_only=args.extract_only)


if __name__=="__main__":
    main()
