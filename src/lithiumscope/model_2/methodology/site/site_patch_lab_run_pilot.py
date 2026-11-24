"""Separated run_pilot responsibility; preserve prior public API."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lithiumscope.model_2.methodology.site_patch_lab import Path

def run_pilot(root:Path,out:Path, *, with_central=True) -> dict:
    from lithiumscope.model_2.methodology.site_patch_lab import ARMS, HISTORICAL_N, LABEL_POLICY, PARAMS, Path, SITE_THRESHOLD, aggregate_training_sites, bootstrap_group_delta, define_outer_fold, describe_sample_subsets, finite_quality_filter, fit_classifier, half_patch_predictors, irreducible_identical_input_minority, json, np, pd, score_metrics, site_level_evaluation, time, validate_inputs
    contract,X,folds,qc,baseline,hashes,source_hashes=validate_inputs(root)
    out.mkdir(parents=True,exist_ok=True)
    design={
        "experiment":"Model2_phase_E_site_patch_quality_pilot_20261112",
        "arms":list(ARMS),"parameters":PARAMS,
        "reference":"Archived phase C legacy_all predictions, unchanged",
        "outer_test_indices":"706 frozen historical rows across 5 folds",
        "all_test_cases_same":True,"fold_li_label_rule":LABEL_POLICY,
        "new_label_policies":"TRAIN ONLY; hypothetical site definitions, not certified",
        "image_quality_threshold":SITE_THRESHOLD,
        "image_quality_applied_to_train_only":True,
        "central_patch_pixels":"[8:24,8:24] from original 32x32 6-band patches",
        "central_features":126,"central_transform_applied_to_all_test_rows":True,
        "no_parameter_tuning":True,
        "known_demo_cases_used":False,
        "training_source_sha256":hashes,
        "existing_evidence_sha256":source_hashes,
        "not_production_calibration":True,
    }
    (out/"experiment_design.json").write_text(
        json.dumps(design,indent=2,ensure_ascii=False),encoding="utf-8")
    center=None
    if with_central:
        t=time.monotonic()
        center=half_patch_predictors(
            contract,list(X.columns),qc,
            progress=lambda message:print(message,flush=True))
        center.to_csv(out/"central_16x16_features.csv",index=False)
        print("CENTER_COMPLETE seconds",round(time.monotonic()-t,2),flush=True)
    all_rows=[]
    per_fold=[]
    inventory=[]
    baseline_copy=baseline.copy()
    baseline_copy["arm"]=ARMS[0]
    baseline_copy["train_unit_count"]=baseline_copy["n_train"]
    baseline_copy["training_source_samples"]=baseline_copy["n_train"]
    baseline_copy["site_policy"]="UNCHANGED"
    all_rows.extend(baseline_copy.to_dict("records"))
    for arm in ARMS[1:]:
        t=time.monotonic()
        rows=[]
        for fidx,fold in enumerate(folds,1):
            _,tr,te,threshold=define_outer_fold(fold,contract)
            original_train=len(tr)
            features=center if arm=="central_16x16_patch" else X
            if features is None:
                raise ValueError("Central feature table was not generated")
            if arm in ("quality_95_train","site_median_quality_95"):
                tr=finite_quality_filter(tr,qc)
            if arm in ("site_median_li","site_majority_consensus","site_median_quality_95"):
                policy=("majority" if arm=="site_majority_consensus" else "median")
                trainX,y_train,site_inventory=aggregate_training_sites(
                    contract,features,tr,
                    lithium_threshold=threshold,policy=policy,
                    sample_sha256=qc.image_sha256)
                for rec in site_inventory.to_dict("records"):
                    rec.update({"arm":arm,"fold":fidx})
                    inventory.append(rec)
                n_training_units=len(trainX)
            else:
                trainX=features.iloc[tr]
                y_train=(contract.Li_icpms.iloc[tr].to_numpy(float)>=threshold).astype(int)
                n_training_units=len(tr)
            if np.unique(y_train).size!=2:
                raise ValueError(f"Single-class train for {arm} fold {fidx}")
            y_test=(contract.Li_icpms.iloc[te].to_numpy(float)>=threshold).astype(int)
            score=fit_classifier(trainX,y_train,features.iloc[te])
            samplemetrics=score_metrics(y_test,score)
            per_fold.append({
                "arm":arm,"fold":fidx,"n_train_samples_before":original_train,
                "n_train_samples_after_quality":len(tr),
                "n_train_site_units":n_training_units,
                "n_test":len(te),"n_test_high_Li":int(y_test.sum()),
                **samplemetrics
            })
            print("ARM",arm,"FOLD",fidx,"TRAIN",n_training_units,
                  "TEST",len(te),"AUC",round(samplemetrics["auc"],4),
                  "AP",round(samplemetrics["ap"],4),flush=True)
            for idx,yt,prob in zip(te,y_test,score):
                rows.append({
                    "arm":arm,
                    "fold":fidx,
                    "test_index":int(idx),
                    "sample_id":str(contract.sample_id.iloc[idx]),
                    "spatial_group":str(contract.spatial_group.iloc[idx]),
                    "site_screening_id":str(contract.site_screening_id.iloc[idx]),
                    "latitude":float(contract.latitude.iloc[idx]),
                    "li_ppm":float(contract.Li_icpms.iloc[idx]),
                    "fold_li_threshold_ppm":threshold,
                    "y_true":int(yt),
                    "score":float(prob),
                    "n_train":original_train,
                    "n_features":len(features.columns),
                    "train_unit_count":n_training_units,
                    "training_source_samples":len(tr),
                    "site_policy":arm,
                })
        pred=pd.DataFrame(rows).sort_values("test_index")
        if pred.test_index.tolist()!=list(range(HISTORICAL_N)):
            raise AssertionError("Each site-patch arm must retain the complete frozen 706 TEST rows")
        if not np.array_equal(pred.y_true.to_numpy(int),
                              baseline_copy.sort_values("test_index").y_true.to_numpy(int)):
            raise AssertionError("Held-out label changed in Phase E")
        all_rows.extend(rows)
        pred.to_csv(out/f"oof_{arm}.csv",index=False)
        (out/"progress.json").write_text(json.dumps({
            "completed":arm,"state":"running","elapsed_arm_s":time.monotonic()-t
        },indent=2),encoding="utf-8")
        print("ARM_COMPLETED",arm,"SECONDS",round(time.monotonic()-t,2),flush=True)
    pred=pd.DataFrame(all_rows)
    pred.to_csv(out/"all_arms_paired_oof.csv",index=False)
    pd.DataFrame(per_fold).to_csv(out/"fold_metrics.csv",index=False)
    if inventory:
        pd.DataFrame(inventory).to_csv(out/"train_site_aggregation_decisions.csv",index=False)
    (out/"identical_patch_label_error_bounds.csv").write_text(
        irreducible_identical_input_minority(contract,folds,qc).to_csv(index=False),
        encoding="utf-8")
    sample_metrics=describe_sample_subsets(pred,qc)
    sample_metrics.to_csv(out/"sample_quality_site_subgroups.csv",index=False)
    site_metrics=site_level_evaluation(pred)
    site_metrics.to_csv(out/"site_level_metrics.csv",index=False)
    summary=[]
    for arm,sub in pred.groupby("arm",sort=False):
        record={"arm":arm,**score_metrics(sub.y_true,sub.score)}
        record["n_sites_in_test"]=int(sub.site_screening_id.nunique())
        record["n_test_high"]=int(sub.y_true.sum())
        summary.append(record)
    pd.DataFrame(summary).to_csv(out/"paired_model_comparison.csv",index=False)
    # Group-bootstrapped exploratory paired effects, not a confirmation test.
    contrast=[]
    for arm in ARMS[1:]:
        result=bootstrap_group_delta(
            pred,ARMS[0],arm,repeats=320,seed=20261112)
        contrast.append(result)
    pd.DataFrame(contrast).to_csv(out/"group_bootstrap_deltas.csv",index=False)
    full={
        "state":"complete",
        "frozen_historical_test_rows":HISTORICAL_N,
        "frozen_outer_folds":5,"arms":list(ARMS),
        "baseline_reused_no_retrain":True,
        "new_fit_count":5*(len(ARMS)-1),
        "data_sources_sha256":hashes,
        "diagnostic_only":True,"model_activation":False,
        "known_demo_cases_read":False,"external_validation":False,
        "scientific_warning":"No candidate may be selected on these historical outer folds; Phase E is exploratory."
    }
    (out/"experiment_summary.json").write_text(
        json.dumps(full,ensure_ascii=False,indent=2),encoding="utf-8")
    (out/"progress.json").write_text(
        json.dumps({"state":"complete","arms":list(ARMS)},indent=2),encoding="utf-8")
    print("PHASE_E_COMPLETE",out,flush=True)
    return full
