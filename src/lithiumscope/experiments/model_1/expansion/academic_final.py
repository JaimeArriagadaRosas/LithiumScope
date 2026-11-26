"""Bounded final academic comparison: enriched CatBoost and local analogs.

The historical evaluation cohort and all five outer folds remain unchanged.
Selection uses three inner folds. No model or production schema is saved.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from uuid import uuid4

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.paths import RESULTS_DIR
from lithiumscope.core.scientific.regression_metrics import regression_metrics
from lithiumscope.datasets.management.local_sources import local_training_source
from lithiumscope.experiments.model_1.balanced.academic import cohort
from lithiumscope.experiments.model_1.expansion.academic_analog import fit_analog
from lithiumscope.experiments.model_1.expansion.academic_bridge import PILOT, SEED, PROFILES, check_pilot, benchmark_status
from lithiumscope.experiments.model_1.expansion.academic_features import RECOVERY_SHA, load_recovery, measurement_blocks, select_columns
from lithiumscope.experiments.model_1.expansion.academic_treatments import fit_profile
from lithiumscope.experiments.model_1.expansion.sensitivity_data import read
from lithiumscope.model_1.schema import validate_training_schema
from lithiumscope.model_1.training.factory import build_pipeline
from lithiumscope.model_1.training.preprocessing.tabular import FeatureSchema
from lithiumscope.resources.detection.device import detect_device
from lithiumscope.resources.manager import estimator_thread_scope
from lithiumscope.results.atomic_exports import write_table, write_legacy_evidence_json
from lithiumscope.results.reference_benchmarks import OASIS_M1

FEATURE_RUN="20261126T163711_3f9dd35a"
FEATURE_SHA="ecb7314e6a4b7ad370d715af7e326c3bd557c06e3284655ec19f3c9f022a044f"
BASE="hgb_combined"
POLICIES={
    BASE:(1.,0.,0.),
    "catboost_combined":(0.,1.,0.),
    "local_analog":(0.,0.,1.),
    "hgb_catboost":(.5,.5,0.),
    "hgb_analog":(.5,0.,.5),
    "catboost_analog":(0.,.5,.5),
    "three_way":(1/3,1/3,1/3),
}
SELECTED="inner_selected"


def combine(hgb,catboost,analog):
    arrays=[np.asarray(v,float) for v in (hgb,catboost,analog)]
    if any(a.ndim!=1 or a.shape!=arrays[0].shape or not np.isfinite(a).all()
           or (a<0).any() for a in arrays):
        raise ValueError("Invalid component predictions")
    return {name:sum(w*a for w,a in zip(weights,arrays,strict=True))
            for name,weights in POLICIES.items()}


def choose(scores):
    if (set(scores.policy)!=set(POLICIES) or scores.duplicated(["policy","inner_fold"]).any()
            or any(set(g.inner_fold)!={1,2,3} for _,g in scores.groupby("policy"))
            or not np.isfinite(scores[["rmse","r2"]]).all().all()):
        raise ValueError("Incomplete inner scores")
    agg=scores.groupby("policy")[["rmse","r2"]].mean()
    eligible=agg.loc[agg.r2.gt(OASIS_M1["r2_mean"])]
    if eligible.empty:
        return BASE
    return str(eligible.reset_index().sort_values(["rmse","r2","policy"],
        ascending=[True,False,True]).iloc[0].policy)


def load_reference(x,pilot,digest):
    root=RESULTS_DIR/"generalization/model_1_expansion_academic_features"/FEATURE_RUN
    if file_sha256(root/"manifest.json")!=FEATURE_SHA:
        raise ValueError("Feature experiment manifest differs")
    m=read(root/"manifest.json")
    if m["state"]!="complete" or m["source_sha256"]!=digest or m["recovery_manifest_sha256"]!=RECOVERY_SHA:
        raise ValueError("Feature experiment identity differs")
    for name,sha in m["outputs"].items():
        if file_sha256(root/name)!=sha:
            raise ValueError("Feature experiment evidence differs: "+name)
    ref=pd.read_csv(root/"predictions.csv",dtype={"row_id":str})
    ref=ref.loc[ref.policy.eq("combined")].sort_values("row_id").reset_index(drop=True)
    if (not ref.row_id.is_unique or not ref[["row_id","fold","group"]].equals(pilot[["row_id","fold","group"]])
            or not np.allclose(ref.truth,pilot.truth,rtol=0,atol=1e-8)
            or not np.isfinite(ref.prediction).all() or (ref.prediction<0).any()):
        raise ValueError("Reference identities differ")
    ref=ref.set_index("row_id").reindex(x.index.astype(str))
    trace={int(t["fold"]):t["features"] for t in read(root/"fit_trace.json")
           if t["stage"]=="outer" and t["policy"]=="combined"}
    if set(trace)!=set(range(1,6)):
        raise ValueError("Missing archived HGB feature trace")
    return ref,trace


def run(*,allow_unverified_metadata=False):
    if not allow_unverified_metadata:
        raise ValueError("Explicit --allow-unverified-metadata required; research only")
    source=local_training_source("model_1")
    digest=file_sha256(source)
    prepared,keep,domain=cohort(source)
    x,y,groups,frame=(v.loc[keep] for v in (prepared.x,prepared.y,prepared.groups,prepared.frame))
    if len(x)!=3263 or len(prepared.x)!=3420 or len(x.columns)!=33:
        raise ValueError("Frozen cohort differs")
    splits=list(KFold(5,shuffle=True,random_state=SEED).split(x))
    pilot,_=check_pilot(RESULTS_DIR/"generalization/model_1_oasis_like"/PILOT,x,y,groups,splits,digest,domain)
    ref,archive_features=load_reference(x,pilot,digest)
    r,m=load_recovery(prepared,digest)
    if not r.academic_member.eq(keep).all():
        raise ValueError("Recovery membership differs")
    measured,ratios=measurement_blocks(frame,r.loc[keep,m["measurement_columns"]],x)
    numeric,categorical=list(prepared.schema.numeric),list(prepared.schema.categorical)
    validate_training_schema(prepared.target,numeric+list(measured)+list(ratios),categorical)
    features=pd.concat([x,measured,ratios],axis=1)
    out=RESULTS_DIR/"generalization/model_1_expansion_academic_final"/(
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")+"_"+uuid4().hex[:8])
    out.mkdir(parents=True,exist_ok=False)
    manifest={"state":"running","scope":"bounded_academic_family_and_analog_comparison",
        "source_sha256":digest,"feature_manifest_sha256":FEATURE_SHA,"recovery_manifest_sha256":RECOVERY_SHA,
        "domain":domain,"rows":len(x),"policies":POLICIES,"oasis_reference":OASIS_M1,
        "planned_tree_fits":35,"planned_analog_fits":20,"archived_outer_hgb_reused":True,
        "random_cv_spatial_overlap":True,"retrospective_target_assisted_record_linkage":True,
        "independent_holdout":False,"direct_comparability":False,"metadata_certified":False,
        "new_training_rows":0,"release_eligible":False,"promotion":False,"models_saved":False}
    write_legacy_evidence_json(out/"manifest.json",manifest)
    device=detect_device(prefer_gpu=False)
    tree_fits=analog_fits=0
    predictions,metrics,inner_scores,trace,neighbors,choices=[],[],[],[],[],[]
    def fit(tr,te,fold,stage):
        nonlocal tree_fits,analog_fits
        if np.intersect1d(tr,te).size:
            raise ValueError("Evaluation overlaps TRAIN")
        extra=select_columns(measured.iloc[tr],groups.iloc[tr])+select_columns(ratios.iloc[tr],groups.iloc[tr])
        schema=FeatureSchema(numeric+extra,categorical,2,"stable_v1")
        cols=schema.numeric+schema.categorical
        tx,vx=features.iloc[tr][cols],features.iloc[te][cols]
        if stage=="outer":
            if cols!=archive_features[fold] or not ref.iloc[te].fold.eq(fold).all():
                raise ValueError("Archived outer HGB identity or features differ")
            h=ref.iloc[te].prediction.to_numpy(float)
        else:
            hfun,info=fit_profile(tx,y.iloc[tr],pd.Series(False,index=tx.index),schema,device,SEED,"native_fine")
            tree_fits+=1
            h=hfun(vx)
            trace.append({"fold":fold,"stage":stage,"model":BASE,"features":cols,**info})
        params={k:v for k,v in PROFILES["catboost"].items() if k!="treatment_policy"}
        model=build_pipeline("catboost",device,SEED,schema,params)
        model.set_params(preprocess__numeric="passthrough")
        with estimator_thread_scope():
            model.fit(tx,y.iloc[tr])
        tree_fits+=1
        c=np.maximum(np.asarray(model.predict(vx),float),0.)
        trace.append({"fold":fold,"stage":stage,"model":"catboost_combined","features":cols,
                      "params":params,"numeric_missing":"native","train_rows":len(tr)})
        analog,info=fit_analog(x.iloc[tr],y.iloc[tr])
        a,nt=analog(x.iloc[te],h)
        analog_fits+=1
        trace.append({"fold":fold,"stage":stage,"model":"local_analog","train_rows":len(tr),**info})
        allowed=set(x.index[tr].astype(str))
        if any(n["row_id"] not in allowed for t in nt for n in t["neighbors"]):
            raise ValueError("Analog neighbor outside TRAIN")
        neighbors.extend({"fold":fold,"stage":stage,**t} for t in nt)
        return combine(h,c,a)

    try:
        for fold,(tr,te) in enumerate(splits,1):
            scores=[]
            for number,(a,b) in enumerate(KFold(3,shuffle=True,random_state=SEED+fold).split(tr),1):
                values=fit(tr[a],tr[b],fold,f"inner_{number}")
                for policy,pred in values.items():
                    scores.append({"fold":fold,"inner_fold":number,"policy":policy,
                        **regression_metrics(y.iloc[tr[b]],pred).to_dict()})
                print(f"Outer {fold}/5 | inner {number}/3",flush=True)
            inner_scores.extend(scores)
            selected=choose(pd.DataFrame(scores))
            choices.append({"fold":fold,"selected_policy":selected})
            values=fit(tr,te,fold,"outer")
            values[SELECTED]=values[selected]
            for policy,pred in values.items():
                mm=regression_metrics(y.iloc[te],pred).to_dict()
                metrics.append({"fold":fold,"policy":policy,**mm})
                predictions.extend({"fold":fold,"policy":policy,"row_id":str(x.index[i]),
                    "group":str(groups.iloc[i]),"truth":float(y.iloc[i]),"prediction":float(v)}
                    for i,v in zip(te,pred,strict=True))
                print(f"Fold {fold}/5 | {policy} | RMSE={mm['rmse']:.4f} | R2={mm['r2']:.4f}",flush=True)
            write_table(out/"predictions.csv",pd.DataFrame(predictions))
            write_table(out/"fold_metrics.csv",pd.DataFrame(metrics))
            write_table(out/"inner_scores.csv",pd.DataFrame(inner_scores))
            write_legacy_evidence_json(out/"fit_trace.json",trace)
            write_legacy_evidence_json(out/"neighbor_trace.json",neighbors)
            write_legacy_evidence_json(out/"selection.json",choices)
        report=[]
        for policy,part in pd.DataFrame(predictions).groupby("policy"):
            part=part.sort_values("row_id").reset_index(drop=True)
            if (not part.row_id.is_unique or not part[["row_id","fold","group"]].equals(pilot[["row_id","fold","group"]])
                    or not np.allclose(part.truth,pilot.truth,rtol=0,atol=1e-8)):
                raise ValueError("Final OOF identity differs")
            fm=pd.DataFrame([regression_metrics(g.truth,g.prediction).to_dict() for _,g in part.groupby("fold")])
            rmse,r2=float(fm.rmse.mean()),float(fm.r2.mean())
            report.append({"policy":policy,"rows":len(part),"rmse_mean":rmse,"r2_mean":r2,
                "mae_mean":float(fm.mae.mean()),**benchmark_status(rmse,r2)})
        if tree_fits!=35 or analog_fits!=20 or len(report)!=8 or file_sha256(source)!=digest:
            raise ValueError("Final campaign incomplete")
        write_table(out/"summary.csv",pd.DataFrame(report))
        manifest["summary"]=report
        manifest["state"]="complete"
    except BaseException as exc:
        manifest["state"]="interrupted"
        manifest["error"]=str(exc)
        raise
    finally:
        manifest["tree_fits"]=tree_fits
        manifest["analog_fits"]=analog_fits
        manifest["outputs"]={p.name:file_sha256(p) for p in out.iterdir() if p.is_file() and p.name!="manifest.json"}
        write_legacy_evidence_json(out/"manifest.json",manifest)
    print(pd.DataFrame(report).to_string(index=False))
    print(f"Estado: {manifest['state']} | Evidencias: {out}",flush=True)
    return out


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-unverified-metadata",action="store_true")
    run(allow_unverified_metadata=parser.parse_args().allow_unverified_metadata)
