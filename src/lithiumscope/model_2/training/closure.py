"""Manual-only Model 2 closure training; explicitly experimental and no release.

The interactive option '2. Modelo 2' calls this runner. Train CatBoost on
the original 706 historical samples (690 with >=95% valid spectral pixels).
Evaluate ALL original 706 samples using the same five spatial outer folds,
with fold-specific Li q75 computed exclusively from historical TRAIN.
Compare to previous-released CatBoost protocol retrained with same folds and
unfiltered historical training samples. Never score the fitted old-release
artifact in-sample and call it CV. Reuse archived phase E OOF as extra control.

No GEOROC provisional labels, external-release claims, activation, Optuna,
new imagery downloads, manual-menu changes, or ten-case selection.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from lithiumscope.core.paths import PROJECT_ROOT
from lithiumscope.core.hashing import file_sha256
from lithiumscope.model_2.methodology.ablations import (
    PARAMS, score_metrics, bootstrap_group_delta, HISTORICAL_N,
    define_outer_fold
)
from lithiumscope.model_2.methodology.site_patch_lab import (
    validate_inputs, finite_quality_filter
)
from lithiumscope.model_2.methodology.three_zone_ranking import ranking_evaluation
from lithiumscope.model_2.training.closure_report import write_closure_report

LIFECYCLE="experimental_not_for_individual_release"
OLD_DISCARDED_ID="training_20261111_170832_m0300"
OLD_RELEASE_ID="training_20260920_195933_m0300"
OLD_RELEASE="models/model_2/trained/"+OLD_RELEASE_ID+"/model.joblib"
HIST_MANIFEST="data/processed/model_2/training_manifest.csv"
ARCHIVED="results/scientific_simulations/m2_phase_e_20261112/pilot/all_arms_paired_oof.csv"
QUALITY_CUTOFF=.95


def _model_from_params(train:pd.DataFrame,labels:np.ndarray,
                       test:pd.DataFrame,params:dict)->np.ndarray:
    from catboost import CatBoostClassifier
    model=CatBoostClassifier(**params)
    model.fit(train,labels)
    return np.asarray(model.predict_proba(test)[:,1],dtype=float)


def prior_release_parameters(bundle:dict)->dict:
    model=bundle.get("estimator")
    if model is None or model.__class__.__name__!="CatBoostClassifier":
        raise ValueError("Previous release M2 is not a CatBoostClassifier")
    params=dict(model.get_params())
    for key in (
        "use_best_model","early_stopping_rounds","od_type","od_wait",
        "cat_features","embedding_features","text_features",
        "eval_metric","custom_metric"
    ):
        params.pop(key,None)
    if params.get("task_type","CPU")!="CPU":
        raise ValueError("Old release protocol must be CPU reproducible")
    params.update({"task_type":"CPU","thread_count":4,
                   "allow_writing_files":False,"verbose":False})
    return params


def prepare(root:Path=PROJECT_ROOT):
    contract,features,folds,qa,baseline,hashes,selected=validate_inputs(root)
    if len(contract)!=783 or features.shape!=(783,126):
        raise ValueError("Experimental dataset size or feature shape changed")
    from lithiumscope.core.config import load_config
    from lithiumscope.model_2.schema import feature_range_profile
    cfg=load_config("model_2")
    source=root/HIST_MANIFEST
    if file_sha256(source)!=cfg["training"]["frozen_manifest_sha256"]:
        raise ValueError("Original 706-row historical manifest changed")
    original=pd.read_csv(source)
    if len(original)!=706 or list(original.sample_id)!=list(contract.sample_id.iloc[:706]):
        raise ValueError("Historical 706 sample order changed")
    for col in ("Li_icpms","latitude","longitude","spatial_group","image_path",
                "sentinel_scene_id"):
        if original[col].astype(str).tolist()!=contract[col].iloc[:706].astype(str).tolist():
            raise ValueError("Historical reference field changed: "+col)
    if qa.sample_id.iloc[:706].tolist()!=original.sample_id.tolist():
        raise ValueError("QA no longer aligns with 706 historical rows")
    eligible=finite_quality_filter(np.arange(706),qa,minimum_fraction=QUALITY_CUTOFF)
    if len(eligible)!=690:
        raise ValueError("Unexpected QA-eligible historical training count")
    original_release=root/OLD_RELEASE
    if not original_release.is_file():
        raise FileNotFoundError("Old M2 release model artifact missing")
    bundle=joblib.load(original_release)
    if list(bundle.get("features",[]))!=list(features.columns):
        raise ValueError("Published model and corrected training feature schema differ")
    old_params=prior_release_parameters(bundle)
    archived=pd.read_csv(root/ARCHIVED)
    for arm in ("historical_706_reference","quality_95_train"):
        ref=archived[archived.arm==arm].sort_values("test_index")
        if len(ref)!=706 or ref.test_index.tolist()!=list(range(706)):
            raise ValueError("Archived reference OOF does not match 706 held-outs")
    fold_checks=[]
    for k,fold in enumerate(folds,1):
        _,old_train,test,threshold=define_outer_fold(fold,contract)
        filtered=finite_quality_filter(old_train,qa)
        if len(np.intersect1d(filtered,test)) or (
                set(contract.spatial_group.iloc[filtered])
                & set(contract.spatial_group.iloc[test])):
            raise ValueError("Train/test spatial leakage")
        if not (np.all(filtered<706) and np.all(test<706)):
            raise ValueError("Provisional GEOROC or changed test indices")
        yt=(contract.Li_icpms.iloc[filtered].to_numpy(float)>=threshold).astype(int)
        yte=(contract.Li_icpms.iloc[test].to_numpy(float)>=threshold).astype(int)
        if len(set(yt))!=2 or len(set(yte))!=2:
            raise ValueError("Single class outer fold")
        fold_checks.append({
            "fold":k,"train_original":len(old_train),
            "train_filtered":len(filtered),
            "test_unchanged":len(test),
            "train_positive":int(yt.sum()),"test_positive":int(yte.sum()),
            "train_only_li_threshold_ppm":float(threshold),
        })
    if sum(x["test_unchanged"] for x in fold_checks)!=706:
        raise ValueError("Frozen heldout coverage changed")
    info={
        "state":"ready_for_manual_training",
        "lifecycle":LIFECYCLE,
        "old_discarded_experiment":OLD_DISCARDED_ID,
        "old_release_reference":OLD_RELEASE_ID,
        "prior_release_fit_params":old_params,
        "corrected_fixed_fit_params":PARAMS,
        "training_historical_samples":706,
        "excluded_by_quality_from_final_training":16,
        "final_training_samples":690,
        "all_706_heldout_samples_preserved":True,
        "historical_CV_folds":fold_checks,
        "feature_count":126,
        "feature_recipe":"legacy_DN_median_126; no unvalidated radiometric switch",
        "test_uses_original_low_quality_inputs":True,
        "spectral_coverage_rule":QUALITY_CUTOFF,
        "GEOROC_provisional_training_rows":0,
        "selection_source":"Exploratory quality 95%% prior phase; no new outer model selection",
        "new_external_holdout":False,
        "release_authorized":False,
        "old_models_or_main_menu_replaced":False,
        "source_hashes":hashes,
    }
    return info,contract,features,folds,qa,eligible,old_params,archived


def _rows(arm:str,fold:int,test:np.ndarray,labels:np.ndarray,
          scores:np.ndarray,threshold:float,contract:pd.DataFrame):
    for index,label,score in zip(test,labels,scores):
        yield {
            "arm":arm,"test_index":int(index),"fold":int(fold),
            "sample_id":str(contract.sample_id.iloc[index]),
            "site_screening_id":str(contract.site_screening_id.iloc[index]),
            "spatial_group":str(contract.spatial_group.iloc[index]),
            "li_ppm":float(contract.Li_icpms.iloc[index]),
            "fold_li_threshold_ppm":float(threshold),
            "y_true":int(label),"score":float(score),
        }


def _evaluate(root:Path,out:Path,contract:pd.DataFrame,
              features:pd.DataFrame,folds:list,qa:pd.DataFrame,
              old_params:dict,archived:pd.DataFrame):
    records=[];metrics_by_fold=[]
    for n,fold in enumerate(folds,1):
        _,tr_hist,test,threshold=define_outer_fold(fold,contract)
        tr_quality=finite_quality_filter(tr_hist,qa)
        ytest=(contract.Li_icpms.iloc[test].to_numpy(float)>=threshold).astype(int)
        for arm,tr,params in (
            ("original_release_protocol_retrained",tr_hist,old_params),
            ("corrected_quality95",tr_quality,PARAMS),
        ):
            labels=(contract.Li_icpms.iloc[tr].to_numpy(float)>=threshold).astype(int)
            scores=_model_from_params(features.iloc[tr],labels,
                                      features.iloc[test],params)
            fold_score=score_metrics(ytest,scores)
            metrics_by_fold.append({
                "arm":arm,"fold":n,"n_train":len(tr),
                "n_test":len(test),"train_only_li_threshold":threshold,
                **fold_score,
            })
            records.extend(_rows(arm,n,test,ytest,scores,threshold,contract))
            print("M2",arm,"fold",n,"train",len(tr),"test",len(test),
                  "AUC",round(fold_score["auc"],4),
                  "AP",round(fold_score["ap"],4),flush=True)
            pd.DataFrame(records).to_csv(out/"paired_oof_progress.csv",index=False)
    both=pd.DataFrame(records)
    for arm in ("corrected_quality95","original_release_protocol_retrained"):
        d=both[both.arm==arm].sort_values("test_index")
        if d.test_index.tolist()!=list(range(706)):
            raise ValueError("Expected the exact same 706 original test samples")
    # Archived reference for cross-checking fixed corrected implementation.
    archived_ref=archived[archived.arm.isin((
        "historical_706_reference","quality_95_train"))].copy()
    all_oof=pd.concat([both,archived_ref],ignore_index=True)
    all_oof.to_csv(out/"all_reference_paired_oof.csv",index=False)
    pd.DataFrame(metrics_by_fold).to_csv(out/"per_fold_comparison.csv",index=False)
    metrics=[]
    for name,g in all_oof.groupby("arm",sort=False):
        metrics.append({"arm":name,**score_metrics(g.y_true,g.score)})
    metric_table=pd.DataFrame(metrics)
    metric_table.to_csv(out/"metrics_comparison.csv",index=False)
    bootstrap=bootstrap_group_delta(
        both,"original_release_protocol_retrained","corrected_quality95",
        repeats=320,seed=42)
    (out/"uncertainty_vs_prior_release.json").write_text(
        json.dumps(bootstrap,indent=2,ensure_ascii=False,default=str),
        encoding="utf-8")
    three,case_rows=ranking_evaluation(both,triples_per_fold=250,seed=42)
    case_rows.to_csv(out/"three_zone_cases.csv",index=False)
    (out/"three_zone_metrics.json").write_text(
        json.dumps(three,indent=2,ensure_ascii=False),encoding="utf-8")
    same=all_oof[all_oof.arm.isin((
        "quality_95_train","corrected_quality95"))]
    prior=all_oof[all_oof.arm=="quality_95_train"].sort_values("test_index")
    curr=all_oof[all_oof.arm=="corrected_quality95"].sort_values("test_index")
    max_difference=float(np.max(np.abs(
        prior.score.to_numpy(float)-curr.score.to_numpy(float))))
    if max_difference>1e-10:
        print("NOTE: corrected run differs numerically from archived QA pilot",
              max_difference,flush=True)
    result={
        "metrics":{r["arm"]:r for r in metrics},
        "paired_uncertainty":bootstrap,
        "three_zone":three,
        "max_difference_vs_archived_QA_pilot":max_difference,
        "same_706_test_cases":True,
        "old_release_method_retrained_on_same_folds":True,
        "old_release_fitted_artifact_not_used_for_in_sample_metrics":True,
        "multiple_comparisons_exploratory":True,
    }
    (out/"evaluation_summary.json").write_text(
        json.dumps(result,indent=2,ensure_ascii=False,default=str),
        encoding="utf-8")
    return result


def train(root:Path=PROJECT_ROOT, *,preflight_only=False,
          smoke=False,output:Path|None=None):
    print("Modelo 2: verificando 783 hashes, 706 muestras historicas, "
          "5 folds y correspondencia de caracteristicas...",flush=True)
    info,contract,X,folds,qa,eligible,old_params,archived=prepare(root)
    print("Preflight OK: 690 filas de train elegibles, "
          "706 test originales, GEOROC provisional excluido.",flush=True)
    if preflight_only:
        if output is not None:
            output.mkdir(parents=True,exist_ok=True)
            (output/"preflight.json").write_text(
                json.dumps(info,indent=2,ensure_ascii=False,default=str),
                encoding="utf-8")
        return info
    if smoke:
        dest=output or root/"results/scientific_simulations/model2_manual_closure_smoke"
        dest.mkdir(parents=True,exist_ok=True)
        _,tr,test,threshold=define_outer_fold(folds[0],contract)
        tr=finite_quality_filter(tr,qa)[:125]
        test=test[:16]
        y=(contract.Li_icpms.iloc[tr].to_numpy(float)>=threshold).astype(int)
        if len(set(y))!=2:raise ValueError("Smoke subset lacks 2 classes")
        scored=_model_from_params(X.iloc[tr],y,X.iloc[test],PARAMS)
        report={"state":"smoke_only","candidate_created":False,
                "n_train":len(tr),"test":len(test),
                "finite_scores":bool(np.isfinite(scored).all()),
                "no_release":True}
        (dest/"smoke.json").write_text(json.dumps(report,indent=2),
                                       encoding="utf-8")
        return report
    from lithiumscope.core.run_context import RunContext
    from lithiumscope.core.states import RunState
    from lithiumscope.core.training_logs import register_training_run
    from lithiumscope.model_2.schema import feature_range_profile
    from lithiumscope.persistence.save_model import save_model_bundle
    from catboost import CatBoostClassifier
    context=RunContext.create("model_2",resume=False)
    out=output or context.root
    out.mkdir(parents=True,exist_ok=True)
    context.tracker.set_state(RunState.RUNNING)
    (out/"manual_closure_preflight.json").write_text(
        json.dumps(info,indent=2,ensure_ascii=False,default=str),
        encoding="utf-8")
    print("MODELO 2: entrenando prototipo experimental; NO RELEASE",flush=True)
    print("Calidad 95% SOLO TRAIN; los 706 TEST históricos se preservan.",flush=True)
    try:
        evaluated=_evaluate(root,out,contract,X,folds,qa,old_params,archived)
    except Exception as exc:
        context.tracker.set_state(RunState.FAILED,failure=str(exc))
        raise
    final_threshold=float(np.quantile(contract.Li_icpms.iloc[:706].to_numpy(float),.75))
    y=(contract.Li_icpms.iloc[eligible].to_numpy(float)>=final_threshold).astype(int)
    model=CatBoostClassifier(**PARAMS)
    model.fit(X.iloc[eligible],y)
    bundle={
        "estimator":model,"features":list(X.columns),
        "band_names":["red","green","blue","nir","swir16","swir22"],
        "normalize_per_band":False,
        "algorithm":"catboost",
        "threshold_ppm":final_threshold,"target_quantile":.75,
        "operating_threshold":.5,
        "operating_threshold_objective":"rank_only_fixed_threshold",
        "feature_extractor_version":"legacy_126_DN_median",
        "spectral_representation":"legacy_DN_median",
        "quality_min_joint_valid":.95,
        "experimental_only":True,
        "ranking_only_not_deposit_probability":True,
        "applicability_profile":feature_range_profile(X.iloc[eligible]),
    }
    meta={
        "model_group":"model_2","algorithm":"catboost",
        "experiment_state":LIFECYCLE,"not_for_release":True,
        "activation_forbidden":True,
        "previous_discarded_experiment":OLD_DISCARDED_ID,
        "previous_release_reference":OLD_RELEASE_ID,
        "historical_samples":706,"trained_eligible":len(eligible),
        "excluded_low_coverage_from_training":16,
        "heldout_unchanged":706,
        "georoc_provisional_training_count":0,
        "Li_cutoff_source":"historical_all_706_q75",
        "params":PARAMS,
        "spectral_representation":"legacy_DN_median_126",
        "evaluation_summary":evaluated,
        "scope":"three-zone ranking research; not release or economic deposit estimate",
    }
    model_path,metadata_path=save_model_bundle(
        "model_2","experimental_model2_corrected_quality95",
        bundle,meta,
        run_id=context.run_id,config_name="model_2",
        feature_schema={
            "features":list(X.columns),
            "representation":"legacy_DN_median_126",
            "quality_min_joint_valid":.95,
        },activate=False,
    )
    status_file=model_path.parent/"candidate_status.json"
    status=json.loads(status_file.read_text(encoding="utf-8"))
    status.update({
        "status":"experimental_awaiting_review",
        "not_for_release":True,
        "eligible_for_individual_publication":False,
        "candidate_discardable":True,
    })
    status_file.write_text(json.dumps(
        status,indent=2,ensure_ascii=False),encoding="utf-8")
    assessment=write_closure_report(
        out,info,evaluated,model_path=model_path)
    # Preserve standard metric-center access without claiming a six-model
    # competition: this is an explicitly labelled 2-protocol comparison.
    metrics_file=out/"metrics_comparison.csv"
    if metrics_file.is_file():
        from shutil import copyfile
        copyfile(metrics_file,context.tables/"experimental_protocol_comparison.csv")
    report={
        "state":"completed_manual_experimental_only",
        "model_path":str(model_path),
        "metadata_path":str(metadata_path),
        "run_dir":str(out),
        "active_release_untouched":True,
        "training_count":len(eligible),
        "test_count":706,"GEOROC_provisional_count":0,
        "new_release_created":False,
        "comparison":evaluated,
        "descriptive_internal_acceptance":assessment["passes_descriptive_internal_criteria"],
        "report_path":str(out/"INFORME_ENTRENAMIENTO_MODELO2_EXPERIMENTAL.md"),
    }
    (out/"manual_training_summary.json").write_text(
        json.dumps(report,indent=2,ensure_ascii=False,default=str),
        encoding="utf-8")
    context.tracker.set_state(
        RunState.COMPLETED,winner="experimental_catboost_quality95",
        release_gate_pass=False,experimental_only=True,
        primary_metric="roc_auc",
        primary_metric_value=float(evaluated["metrics"]["corrected_quality95"]["auc"]),
        manual_closure_accepted_internally=assessment["passes_descriptive_internal_criteria"],
        model_path=str(model_path))
    register_training_run("model_2",out)
    print("EXPERIMENTAL_M2_COMPLETE",str(out),flush=True)
    print("NO RELEASE; NO ACTIVE MODEL CHANGE",flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preflight",action="store_true")
    p.add_argument("--smoke",action="store_true")
    p.add_argument("--output")
    args=p.parse_args()
    out=Path(args.output) if args.output else None
    if out is not None and not out.is_absolute():out=PROJECT_ROOT/out
    result=train(PROJECT_ROOT,preflight_only=args.preflight,smoke=args.smoke,
                 output=out)
    print(json.dumps({k:v for k,v in result.items()
                      if not isinstance(v,(dict,list))},
                     indent=2,ensure_ascii=False,default=str))

if __name__=="__main__":
    main()
