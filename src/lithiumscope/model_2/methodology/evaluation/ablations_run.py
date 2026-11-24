"""Separated run responsibility; preserve prior public API."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lithiumscope.model_2.methodology.ablations import Path

def run(root:Path,output:Path, *, nested=False):
    from lithiumscope.model_2.methodology.ablations import ARMS, HISTORICAL_N, PARAMS, Path, bootstrap_group_delta, build_inputs, define_outer_fold, fit_classifier, json, nested_decision_probe, pd, score_metrics, select_train_indices, site_weights, time
    contract,predictors,folds,raw_names,hashes=build_inputs(root)
    output.mkdir(parents=True,exist_ok=True)
    configuration={
        "schema_version":1,"scope":"Phase C fixed ablations; historical holdout only",
        "arms":list(ARMS),"params":PARAMS,
        "historical_test_rows":HISTORICAL_N,
        "feature_count_full":len(predictors.columns),
        "feature_count_raw":len(raw_names),
        "raw_feature_columns":raw_names,
        "folds":len(folds),"random_seed":42,
        "known_ten_cases_read":False,"new_sources_downloaded":False,
        "source_sha256":hashes,
        "no_hyperparameter_selection":True,
        "no_model_activation":True,"no_final_refit":True,
        "outer_threshold_provenance":"q75 historical TRAIN per fold",
        "warning":"Outer results exploratory for arm development; no confirmatory holdout.",
    }
    (output/"experiment_design.json").write_text(
        json.dumps(configuration,indent=2,ensure_ascii=False),encoding="utf-8")
    allrows=[]
    completed={}
    for arm in ARMS:
        t0=time.monotonic()
        arm_rows=[]
        for k,fold in enumerate(folds,1):
            base,legacy,te,thr=define_outer_fold(fold,contract)
            tr=select_train_indices(arm,fold,contract,threshold=thr)
            columns=raw_names if arm=="expanded_raw_spectral_only" else list(predictors.columns)
            y_tr=(contract.Li_icpms.iloc[tr].to_numpy(float)>=thr).astype(int)
            y_te=(contract.Li_icpms.iloc[te].to_numpy(float)>=thr).astype(int)
            weights=site_weights(contract,tr) if arm=="expanded_inverse_site_weights" else None
            score=fit_classifier(predictors.iloc[tr][columns],y_tr,
                                 predictors.iloc[te][columns],weights=weights)
            fold_metrics=score_metrics(y_te,score)
            print("ABLATION",arm,"fold",k,"train",len(tr),"historical_test",len(te),
                  "auc",round(fold_metrics["auc"],4) if fold_metrics["auc"] is not None else None,
                  "ap",round(fold_metrics["ap"],4) if fold_metrics["ap"] is not None else None,
                  flush=True)
            for idx,y,s in zip(te,y_te,score):
                arm_rows.append({"arm":arm,"fold":k,"test_index":int(idx),
                    "sample_id":str(contract.sample_id.iloc[idx]),
                    "spatial_group":str(contract.spatial_group.iloc[idx]),
                    "site_screening_id":str(contract.site_screening_id.iloc[idx]),
                    "latitude":float(contract.latitude.iloc[idx]),
                    "li_ppm":float(contract.Li_icpms.iloc[idx]),
                    "fold_li_threshold_ppm":float(thr),
                    "y_true":int(y),"score":float(s),
                    "n_train":len(tr),"n_features":len(columns)})
        df=pd.DataFrame(arm_rows).sort_values("test_index")
        if df.test_index.tolist()!=list(range(HISTORICAL_N)):
            raise AssertionError("Each arm must score the SAME 706 historical cases once")
        allrows.extend(arm_rows)
        df.to_csv(output/("oof_"+arm+".csv"),index=False)
        completed[arm]={"seconds":time.monotonic()-t0,**score_metrics(df.y_true,df.score)}
        (output/"run_progress.json").write_text(json.dumps({
            "state":"running","finished_arms":completed,"expected_arms":list(ARMS)
        },indent=2),encoding="utf-8")
        print("ARM_DONE",arm,"pooled",completed[arm],flush=True)
    pred=pd.DataFrame(allrows)
    pred.to_csv(output/"all_oof_arms.csv",index=False)
    rows=[]
    for arm,df in pred.groupby("arm",sort=False):
        for fold,sub in df.groupby("fold",sort=False):
            rows.append({"arm":arm,"fold":int(fold),"n_train":int(sub.n_train.iloc[0]),
                         "n_features":int(sub.n_features.iloc[0]),**score_metrics(sub.y_true,sub.score)})
    pd.DataFrame(rows).to_csv(output/"fold_metrics.csv",index=False)
    rank=[]
    for arm,info in completed.items():
        reference=completed["expanded_all"] if arm!="legacy_all" else completed["legacy_all"]
        rank.append({"arm":arm,**info,
            "delta_auc_vs_expanded":info["auc"]-completed["expanded_all"]["auc"],
            "delta_ap_vs_expanded":info["ap"]-completed["expanded_all"]["ap"],
            "delta_brier_vs_expanded":info["brier"]-completed["expanded_all"]["brier"]})
    ranking=pd.DataFrame(rank)
    ranking.to_csv(output/"exploratory_comparison.csv",index=False)
    # Fixed baseline to augmented contrast and other single-factor contrasts.
    pairs=[
        ("legacy_all","expanded_all"),
        ("expanded_all","expanded_buffer_5km"),
        ("expanded_all","expanded_buffer_10km"),
        ("expanded_all","expanded_exclude_shared_scenes"),
        ("expanded_all","expanded_drop_mixed_train_sites"),
        ("expanded_all","expanded_raw_spectral_only"),
        ("expanded_all","expanded_inverse_site_weights"),
    ]
    boots=[]
    for ref,arm in pairs:
        boots.append(bootstrap_group_delta(pred,ref,arm,repeats=320))
    pd.DataFrame(boots).to_csv(output/"cluster_bootstrap_exploratory.csv",index=False)
    if nested:
        choices=("legacy_all","expanded_buffer_5km")
        nested_records=[]
        for arm in choices:
            print("NESTED_DECISION_BEGIN",arm,flush=True)
            rules=nested_decision_probe(contract,predictors,folds,arm=arm)
            rulelookup=rules.set_index("fold").decision_threshold
            subset=pred[pred.arm==arm]
            for fold,sub in subset.groupby("fold"):
                scores=sub.score.to_numpy(float)
                metrics=score_metrics(sub.y_true,scores,float(rulelookup.loc[fold]))
                nested_records.append({"arm":arm,"fold":int(fold),
                    "train_only_decision_threshold":float(rulelookup.loc[fold]),
                    **{("decision_"+k):v for k,v in metrics.items()}})
            rules.to_csv(output/("inner_frozen_thresholds_"+arm+".csv"),index=False)
        pd.DataFrame(nested_records).to_csv(output/"inner_threshold_outer_assessment.csv",index=False)
    summary={"state":"completed","results":completed,
             "same_706_held_out":True,"no_activation":True,
             "bootstrap_pairs":[{"reference":r,"arm":v} for r,v in pairs],
             "nested_probe":bool(nested)}
    (output/"experiment_summary.json").write_text(
        json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8")
    (output/"run_progress.json").write_text(
        json.dumps({"state":"completed","finished_arms":completed,
                    "nested_probe":bool(nested)},indent=2),encoding="utf-8")
    print("PHASE_C_COMPLETE",output,flush=True)
    return summary
