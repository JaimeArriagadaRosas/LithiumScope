"""Bounded XGB/CatBoost training-method comparison; no data or release changes."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from uuid import uuid4
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.paths import RESULTS_DIR
from lithiumscope.core.reproducibility import runtime_fingerprint
from lithiumscope.core.scientific.regression_metrics import regression_metrics
from lithiumscope.datasets.management.local_sources import local_training_source
from lithiumscope.experiments.model_1.balanced.academic import cohort
from lithiumscope.experiments.model_1.expansion.academic_bridge import PILOT,SEED,PROFILES,check_pilot,benchmark_status
from lithiumscope.experiments.model_1.expansion.academic_final import load_reference,FEATURE_SHA
from lithiumscope.experiments.model_1.expansion.academic_features import load_recovery,measurement_blocks,select_columns,RECOVERY_SHA
from lithiumscope.experiments.model_1.expansion.sensitivity_data import read
from lithiumscope.experiments.model_1.integral.training_losses import score,summarize_scores,BINS,TAIL,MAE_TOLERANCE
from lithiumscope.model_1.training.preprocessing.tabular import FeatureSchema
from lithiumscope.model_1.training.factory import build_pipeline
from lithiumscope.model_1.schema import validate_training_schema
from lithiumscope.resources.detection.device import detect_device
from lithiumscope.resources.manager import estimator_thread_scope
from lithiumscope.results.atomic_exports import write_table,write_legacy_evidence_json

HGB_RUN="20261126T214831_885e8ea1"
HGB_SHA="93af2f8fa5dbe2bfb07e3c4caa1b30599a5c197f6941487594390326ce901e29"
CAT_RUN="20261126T165331_935bc05f"
CAT_SHA="ec1831183f53a4b25cd905ac742ad06a019d1f39ce7132f13c3a0f09fa8baa66"
BASE="hgb_squared"
CAT="catboost_rmse"
SELECTED="inner_selected"
XGB={"n_estimators":400,"learning_rate":.05,"max_depth":6,"subsample":.9,
     "colsample_bytree":.9,"objective":"reg:squarederror","device":"cpu","tree_method":"hist",
     "reg_alpha":0.,"reg_lambda":1.,"min_child_weight":1.}
CB={**{k:v for k,v in PROFILES["catboost"].items() if k!="treatment_policy"},
    "task_type":"CPU","loss_function":"RMSE"}
PROFILES_NEW={
    "xgb_squared":("xgboost",XGB),
    "xgb_regularized":("xgboost",{**XGB,"reg_alpha":.1,"reg_lambda":5.,"min_child_weight":5.}),
    "xgb_pseudohuber":("xgboost",{**XGB,"objective":"reg:pseudohubererror","huber_slope":10.}),
    "catboost_huber":("catboost",{**CB,"loss_function":"Huber:delta=10"}),
    "catboost_expectile":("catboost",{**CB,"loss_function":"Expectile:alpha=0.65"}),
}
POLICIES=(BASE,CAT,*PROFILES_NEW)
MAX_FITS=100


def archived_run(folder,run,sha,digest):
    root=RESULTS_DIR/"generalization"/folder/run
    if file_sha256(root/"manifest.json")!=sha:
        raise ValueError("Archived manifest differs")
    m=read(root/"manifest.json")
    if (m["state"]!="complete" or m["source_sha256"]!=digest
            or m["feature_manifest_sha256"]!=FEATURE_SHA or m["recovery_manifest_sha256"]!=RECOVERY_SHA):
        raise ValueError("Archived source/feature identity differs")
    for name,expected in m["outputs"].items():
        if file_sha256(root/name)!=expected:
            raise ValueError("Archived output differs: "+name)
    return root,m


def archive_predictions(root,policy,pilot,x):
    p=pd.read_csv(root/"predictions.csv",dtype={"row_id":str})
    p=p.loc[p.policy.eq(policy)].sort_values("row_id").reset_index(drop=True)
    if (not p.row_id.is_unique or not p[["row_id","fold","group"]].equals(pilot[["row_id","fold","group"]])
            or not np.allclose(p.truth,pilot.truth,rtol=0,atol=1e-8)
            or not np.isfinite(p.prediction).all() or p.prediction.lt(0).any()):
        raise ValueError("Archived OOF identity differs")
    return p.set_index("row_id").reindex(x.index.astype(str))


def fit_method(x,y,validation,schema,device,policy):
    if (not x.index.is_unique or not x.index.equals(y.index) or set(x.index)&set(validation.index)
            or not len(x) or not np.isfinite(y).all() or y.le(0).any()):
        raise ValueError("Invalid or overlapping TRAIN")
    cols=schema.numeric+schema.categorical
    if list(x.columns)!=cols or list(validation.columns)!=cols:
        raise ValueError("Feature contract differs")
    family,params=PROFILES_NEW[policy]
    model=build_pipeline(family,device,SEED,schema,params)
    model.set_params(preprocess__numeric="passthrough")
    with estimator_thread_scope():
        model.fit(x,y)
    pred=np.maximum(np.asarray(model.predict(validation),float),0.)
    if pred.shape!=(len(validation),) or not np.isfinite(pred).all():
        raise ValueError("Invalid boosting prediction")
    return pred


def choose(frame):
    # CatBoost archived outer control is descriptive only: its archived
    # inner scores lack tail SSE. Never invent it or use outer truth.
    candidates={BASE,*PROFILES_NEW}
    if (set(frame.policy)!=candidates or frame.duplicated(["policy","inner_fold"]).any()
            or any(set(g.inner_fold)!={1,2,3} for _,g in frame.groupby("policy"))):
        raise ValueError("Incomplete inner comparison")
    stats={p:summarize_scores(g) for p,g in frame.groupby("policy")}
    base=stats[BASE]
    allowed=[p for p,v in stats.items() if p!=BASE and v["rmse_mean"]<base["rmse_mean"]
             and v["r2_mean"]>.499 and v["mae_mean"]<=base["mae_mean"]+MAE_TOLERANCE
             and v["tail_rmse"]<=base["tail_rmse"]]
    return min(allowed,key=lambda p:(stats[p]["rmse_mean"],stats[p]["mae_mean"],p)) if allowed else BASE


def check_partition(part,fold,x,tr,te,cols):
    if (part["fold"]!=fold or part["train_ids"]!=x.index[tr].astype(str).tolist()
            or part["test_ids"]!=x.index[te].astype(str).tolist() or part["features"]!=cols):
        raise ValueError("Archived partition differs")


def run(*,allow_unverified_metadata=False):
    if not allow_unverified_metadata:
        raise ValueError("Explicit --allow-unverified-metadata required; research only")
    source=local_training_source("model_1")
    digest=file_sha256(source)
    prepared,keep,domain=cohort(source)
    x,y,groups,frame=(v.loc[keep] for v in (prepared.x,prepared.y,prepared.groups,prepared.frame))
    if len(prepared.x)!=3420 or len(x)!=3263 or len(x.columns)!=33:
        raise ValueError("Frozen cohort differs")
    splits=list(KFold(5,shuffle=True,random_state=SEED).split(x))
    pilot,_=check_pilot(RESULTS_DIR/"generalization/model_1_oasis_like"/PILOT,x,y,groups,splits,digest,domain)
    reference,archived_cols=load_reference(x,pilot,digest)
    hr,hm=archived_run("model_1_integral_training_losses",HGB_RUN,HGB_SHA,digest)
    cr,cm=archived_run("model_1_expansion_academic_final",CAT_RUN,CAT_SHA,digest)
    hgb=archive_predictions(hr,"squared_error",pilot,x)
    cat=archive_predictions(cr,"catboost_combined",pilot,x)
    if not np.allclose(hgb.prediction,reference.prediction,rtol=0,atol=1e-10):
        raise ValueError("HGB controls differ")
    old_inner=pd.read_csv(hr/"inner_scores.csv")
    old_inner=old_inner.loc[old_inner.policy.eq("squared_error")].assign(policy=BASE)
    if len(old_inner)!=15 or old_inner.duplicated(["fold","inner_fold"]).any():
        raise ValueError("Archived inner control differs")
    old_parts=read(hr/"partitions.json")
    old_trace=read(hr/"fit_trace.json")
    htrace={(v["fold"],v["stage"]):v for v in old_trace if v["policy"]=="squared_error"}
    ctrace={v["fold"]:v for v in read(cr/"fit_trace.json") if v["stage"]=="outer" and v["model"]=="catboost_combined"}
    if len(old_parts)!=5 or len(htrace)!=15 or set(ctrace)!=set(range(1,6)):
        raise ValueError("Archived trace differs")
    recovered,rm=load_recovery(prepared,digest)
    if not recovered.academic_member.eq(keep).all():
        raise ValueError("Recovery membership differs")
    measured,ratios=measurement_blocks(frame,recovered.loc[keep,rm["measurement_columns"]],x)
    features=pd.concat([x,measured,ratios],axis=1)
    numeric,categorical=list(prepared.schema.numeric),list(prepared.schema.categorical)
    validate_training_schema(prepared.target,numeric+list(measured)+list(ratios),categorical)
    out=RESULTS_DIR/"generalization/model_1_integral_boosting_methods"/(
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")+"_"+uuid4().hex[:8])
    out.mkdir(parents=True,exist_ok=False)
    manifest={"state":"running","scope":"fixed_boosting_method_development","runtime":runtime_fingerprint(),
        "source_sha256":digest,"feature_manifest_sha256":FEATURE_SHA,"recovery_manifest_sha256":RECOVERY_SHA,
        "hgb_manifest_sha256":HGB_SHA,"catboost_manifest_sha256":CAT_SHA,"domain":domain,"rows":len(x),
        "new_profiles":PROFILES_NEW,"max_new_fits":MAX_FITS,"archived_outer_controls":[BASE,CAT],
        "archived_inner_control":BASE,"catboost_control_descriptive_only":True,
        "seed":SEED,"outer_folds":5,"inner_folds":3,"reused_development_folds":True,
        "random_cv_spatial_overlap":True,"retrospective_target_assisted_recovery":True,
        "independent_holdout":False,"direct_comparability":False,"metadata_certified":False,
        "new_training_rows":0,"new_predictors":False,"production_schema_changed":False,
        "models_saved":False,"release_eligible":False,"promotion":False}
    write_legacy_evidence_json(out/"manifest.json",manifest)
    device=detect_device(prefer_gpu=False)
    fits=0
    predictions,metrics,inner_scores,trace,choices,partitions=[],[],[],[],[],[]

    def schema_for(tr):
        extra=select_columns(measured.iloc[tr],groups.iloc[tr])+select_columns(ratios.iloc[tr],groups.iloc[tr])
        return FeatureSchema(numeric+extra,categorical,2,"stable_v1")

    def fit(tr,te,policy,fold,stage,schema):
        nonlocal fits
        cols=schema.numeric+schema.categorical
        fits+=1
        pred=fit_method(features.iloc[tr][cols],y.iloc[tr],features.iloc[te][cols],schema,device,policy)
        trace.append({"fold":fold,"stage":stage,"policy":policy,"features":cols,
                      "family":PROFILES_NEW[policy][0],"params":PROFILES_NEW[policy][1],
                      "train_rows":len(tr),"validation_rows":len(te)})
        return pred

    try:
        for fold,(tr,te) in enumerate(splits,1):
            schema=schema_for(tr)
            cols=schema.numeric+schema.categorical
            part=old_parts[fold-1]
            check_partition(part,fold,x,tr,te,cols)
            if cols!=archived_cols[fold] or cols!=ctrace[fold]["features"]:
                raise ValueError("Archived outer feature contract differs")
            inner=[]
            for j,(a,b) in enumerate(KFold(3,shuffle=True,random_state=SEED+fold).split(tr),1):
                it,iv=tr[a],tr[b]
                sub=part["inner"][j-1]
                inner_schema=schema_for(it)
                if (sub["inner_fold"]!=j or sub["train_ids"]!=x.index[it].astype(str).tolist()
                        or sub["validation_ids"]!=x.index[iv].astype(str).tolist()
                        or htrace[fold,f"inner_{j}"]["features"]!=inner_schema.numeric+inner_schema.categorical):
                    raise ValueError("Archived inner feature/partition differs")
                refrow=old_inner.loc[old_inner.fold.eq(fold)&old_inner.inner_fold.eq(j)].iloc[0].to_dict()
                if int(refrow["tail_n"])!=int(y.iloc[iv].gt(TAIL).sum()):
                    raise ValueError("Archived inner tail support differs")
                inner.append(refrow)
                for policy in PROFILES_NEW:
                    pred=fit(it,iv,policy,fold,f"inner_{j}",inner_schema)
                    inner.append({"fold":fold,"inner_fold":j,"policy":policy,**score(y.iloc[iv],pred)})
                print(f"Outer {fold}/5 | inner {j}/3",flush=True)
            inner_scores.extend(inner)
            chosen=choose(pd.DataFrame(inner))
            choices.append({"fold":fold,"selected_policy":chosen})
            partitions.append(part)
            values={BASE:hgb.iloc[te].prediction.to_numpy(float),CAT:cat.iloc[te].prediction.to_numpy(float)}
            for policy in PROFILES_NEW:
                values[policy]=fit(tr,te,policy,fold,"outer",schema)
            values[SELECTED]=values[chosen]
            for policy,pred in values.items():
                mm=score(y.iloc[te],pred)
                metrics.append({"fold":fold,"policy":policy,**mm})
                predictions.extend({"fold":fold,"policy":policy,"row_id":str(x.index[i]),
                    "group":str(groups.iloc[i]),"truth":float(y.iloc[i]),"prediction":float(v)}
                    for i,v in zip(te,pred,strict=True))
                print(f"Fold {fold}/5 | {policy} | RMSE={mm['rmse']:.4f} | R2={mm['r2']:.4f}",flush=True)
            write_table(out/"predictions.csv",pd.DataFrame(predictions))
            write_table(out/"fold_metrics.csv",pd.DataFrame(metrics))
            write_table(out/"inner_scores.csv",pd.DataFrame(inner_scores))
            write_legacy_evidence_json(out/"fit_trace.json",trace)
            write_legacy_evidence_json(out/"selection.json",choices)
            write_legacy_evidence_json(out/"partitions.json",partitions)
        report,bands=[],[]
        for policy,p in pd.DataFrame(predictions).groupby("policy"):
            p=p.sort_values("row_id").reset_index(drop=True)
            if (not p.row_id.is_unique or not p[["row_id","fold","group"]].equals(pilot[["row_id","fold","group"]])
                    or not np.allclose(p.truth,pilot.truth,rtol=0,atol=1e-8)):
                raise ValueError("OOF identity differs")
            fm=pd.DataFrame([score(g.truth,g.prediction) for _,g in p.groupby("fold")])
            agg=summarize_scores(fm)
            tail=p.loc[p.truth.gt(TAIL)]
            gm=regression_metrics(p.truth,p.prediction)
            report.append({"policy":policy,**agg,"rmse_global":gm.rmse,"r2_global":gm.r2,
                "rmse_std":float(fm.rmse.std(ddof=1)),"tail_bias":float((tail.prediction-tail.truth).mean()),
                **benchmark_status(agg["rmse_mean"],agg["r2_mean"])})
            for band,g in p.groupby(pd.cut(p.truth,BINS),observed=True):
                bands.append({"policy":policy,"band":str(band),"rows":len(g),**regression_metrics(g.truth,g.prediction).to_dict()})
        base=next(r for r in report if r["policy"]==BASE)
        if fits!=MAX_FITS or len(report)!=len(POLICIES)+1 or file_sha256(source)!=digest:
            raise ValueError("Boosting comparison incomplete")
        for r in report:
            r["development_improvement"]=bool(r["rmse_mean"]<base["rmse_mean"] and r["r2_mean"]>.499
                and r["mae_mean"]<=base["mae_mean"]+MAE_TOLERANCE and r["tail_rmse"]<=base["tail_rmse"])
        write_table(out/"summary.csv",pd.DataFrame(report))
        write_table(out/"band_metrics.csv",pd.DataFrame(bands))
        manifest["summary"]=report
        manifest["state"]="complete"
    except BaseException as exc:
        manifest["state"]="interrupted"
        manifest["error"]=str(exc)
        raise
    finally:
        manifest["new_fits"]=fits
        manifest["outputs"]={p.name:file_sha256(p) for p in out.iterdir() if p.is_file() and p.name!="manifest.json"}
        write_legacy_evidence_json(out/"manifest.json",manifest)
    print(pd.DataFrame(report).to_string(index=False))
    print(f"Estado: {manifest['state']} | Evidencias: {out}",flush=True)
    return out


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-unverified-metadata",action="store_true")
    run(allow_unverified_metadata=parser.parse_args().allow_unverified_metadata)
