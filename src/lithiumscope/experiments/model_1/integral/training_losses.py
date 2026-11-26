"""Fixed native HGB loss comparison on the frozen academic development cohort."""
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
from lithiumscope.experiments.model_1.expansion.academic_bridge import PILOT, SEED, PROFILES, check_pilot, benchmark_status
from lithiumscope.experiments.model_1.expansion.academic_final import load_reference, FEATURE_SHA
from lithiumscope.experiments.model_1.expansion.academic_features import load_recovery, measurement_blocks, select_columns, RECOVERY_SHA
from lithiumscope.model_1.schema import validate_training_schema
from lithiumscope.model_1.training.preprocessing.tabular import FeatureSchema
from lithiumscope.model_1.training.factory import build_pipeline
from lithiumscope.resources.detection.device import detect_device
from lithiumscope.resources.manager import estimator_thread_scope
from lithiumscope.results.atomic_exports import write_table, write_legacy_evidence_json

BASE = "squared_error"
SELECTED = "inner_selected"
LOSSES = {
    BASE: {"loss": "squared_error"},
    "absolute_error": {"loss": "absolute_error"},
    "gamma": {"loss": "gamma"},
    "poisson": {"loss": "poisson"},
    "quantile_060": {"loss": "quantile", "quantile": .60},
}
MAX_FITS = 95
TAIL = 40.
MAE_TOLERANCE = .1
BINS = [0, 5, 10, 20, 40, 60, 80, 100, np.inf]


def loss_params(policy):
    if policy not in LOSSES:
        raise ValueError("Unknown HGB loss policy")
    params = {k:v for k,v in PROFILES["hist_gradient_boosting"].items() if k!="treatment_policy"}
    return {**params, "min_samples_leaf":5, "early_stopping":False, **LOSSES[policy]}


def fit_loss(x, y, validation, schema, device, policy):
    if (not x.index.is_unique or not x.index.equals(y.index)
            or set(x.index)&set(validation.index) or not len(x)
            or not np.isfinite(y).all() or y.le(0).any()):
        raise ValueError("Invalid or overlapping academic TRAIN/validation")
    cols = schema.numeric+schema.categorical
    if list(x.columns)!=cols or list(validation.columns)!=cols:
        raise ValueError("Feature contract differs")
    params = loss_params(policy)
    model = build_pipeline("hist_gradient_boosting",device,SEED,schema,params)
    model.set_params(preprocess__numeric="passthrough")
    with estimator_thread_scope():
        model.fit(x,y,model__sample_weight=np.ones(len(y)))
    pred = np.maximum(np.asarray(model.predict(validation),float),0.)
    if pred.shape!=(len(validation),) or not np.isfinite(pred).all():
        raise ValueError("Invalid loss prediction")
    return pred


def score(y,pred):
    y=np.asarray(y,float)
    pred=np.asarray(pred,float)
    m=regression_metrics(y,pred).to_dict()
    tail=y>TAIL
    return {**m,"tail_n":int(tail.sum()),"tail_sse":float(np.square(y[tail]-pred[tail]).sum())}


def summarize_scores(frame):
    if frame.empty or not np.isfinite(frame[["rmse","r2","mae","tail_sse"]]).all().all():
        raise ValueError("Incomplete or invalid scores")
    n=int(frame.tail_n.sum())
    if n<=0 or (frame.tail_sse<0).any():
        raise ValueError("No tail support in score population")
    return {"rmse_mean":float(frame.rmse.mean()),"r2_mean":float(frame.r2.mean()),
            "mae_mean":float(frame.mae.mean()),"tail_rmse":float(np.sqrt(frame.tail_sse.sum()/n))}


def select_loss(frame):
    if (set(frame.policy)!=set(LOSSES) or frame.duplicated(["policy","inner_fold"]).any()
            or any(set(g.inner_fold)!={1,2,3} for _,g in frame.groupby("policy"))):
        raise ValueError("Incomplete inner loss comparison")
    stats={p:summarize_scores(g) for p,g in frame.groupby("policy")}
    base=stats[BASE]
    eligible=[p for p,v in stats.items() if p!=BASE and v["rmse_mean"]<base["rmse_mean"]
        and v["r2_mean"]>.499 and v["mae_mean"]<=base["mae_mean"]+MAE_TOLERANCE
        and v["tail_rmse"]<=base["tail_rmse"]]
    return min(eligible,key=lambda p:(stats[p]["rmse_mean"],stats[p]["mae_mean"],p)) if eligible else BASE


def run(*,allow_unverified_metadata=False):
    if not allow_unverified_metadata:
        raise ValueError("Explicit --allow-unverified-metadata required; enriched historical research only")
    source=local_training_source("model_1")
    digest=file_sha256(source)
    prepared,keep,domain=cohort(source)
    x,y,groups,frame=(v.loc[keep] for v in (prepared.x,prepared.y,prepared.groups,prepared.frame))
    if len(prepared.x)!=3420 or len(x)!=3263 or len(x.columns)!=33 or y.le(0).any():
        raise ValueError("Frozen academic cohort differs")
    splits=list(KFold(5,shuffle=True,random_state=SEED).split(x))
    pilot,_=check_pilot(RESULTS_DIR/"generalization/model_1_oasis_like"/PILOT,x,y,groups,splits,digest,domain)
    ref,archive_features=load_reference(x,pilot,digest)
    recovered,rm=load_recovery(prepared,digest)
    if not recovered.academic_member.eq(keep).all():
        raise ValueError("Recovery membership differs")
    measured,ratios=measurement_blocks(frame,recovered.loc[keep,rm["measurement_columns"]],x)
    features=pd.concat([x,measured,ratios],axis=1)
    numeric,categorical=list(prepared.schema.numeric),list(prepared.schema.categorical)
    validate_training_schema(prepared.target,numeric+list(measured)+list(ratios),categorical)
    out=RESULTS_DIR/"generalization/model_1_integral_training_losses"/(
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")+"_"+uuid4().hex[:8])
    out.mkdir(parents=True,exist_ok=False)
    manifest={"state":"running","scope":"fixed_hgb_loss_training_method_development",
        "source_sha256":digest,"feature_manifest_sha256":FEATURE_SHA,"recovery_manifest_sha256":RECOVERY_SHA,
        "domain":domain,"rows":len(x),"losses":LOSSES,"max_new_fits":MAX_FITS,
        "archived_outer_control":True,"seed":SEED,"inner_folds":3,"outer_folds":5,
        "reused_development_folds":True,"random_cv_spatial_overlap":True,
        "retrospective_target_assisted_recovery":True,"independent_holdout":False,
        "metadata_certified":False,"direct_comparability":False,"new_training_rows":0,
        "new_predictors":False,"production_schema_changed":False,"models_saved":False,
        "release_eligible":False,"promotion":False}
    write_legacy_evidence_json(out/"manifest.json",manifest)
    device=detect_device(prefer_gpu=False)
    fits=0
    predictions,metrics,inner_scores,trace,choices,partitions=[],[],[],[],[],[]

    def schema_for(tr):
        extra=select_columns(measured.iloc[tr],groups.iloc[tr])+select_columns(ratios.iloc[tr],groups.iloc[tr])
        return FeatureSchema(numeric+extra,categorical,2,"stable_v1")

    def fit(tr,te,policy,fold,stage,schema):
        nonlocal fits
        if np.intersect1d(tr,te).size:
            raise ValueError("TRAIN and validation overlap")
        cols=schema.numeric+schema.categorical
        fits+=1
        pred=fit_loss(features.iloc[tr][cols],y.iloc[tr],features.iloc[te][cols],schema,device,policy)
        trace.append({"fold":fold,"stage":stage,"policy":policy,"params":loss_params(policy),
                      "features":cols,"train_rows":len(tr),"validation_rows":len(te)})
        return pred

    try:
        for fold,(tr,te) in enumerate(splits,1):
            inner=[]
            part={"fold":fold,"train_ids":x.index[tr].astype(str).tolist(),
                  "test_ids":x.index[te].astype(str).tolist(),"inner":[]}
            for j,(a,b) in enumerate(KFold(3,shuffle=True,random_state=SEED+fold).split(tr),1):
                it,iv=tr[a],tr[b]
                schema=schema_for(it)
                part["inner"].append({"inner_fold":j,"train_ids":x.index[it].astype(str).tolist(),
                                     "validation_ids":x.index[iv].astype(str).tolist()})
                for policy in LOSSES:
                    pred=fit(it,iv,policy,fold,f"inner_{j}",schema)
                    inner.append({"fold":fold,"inner_fold":j,"policy":policy,**score(y.iloc[iv],pred)})
            inner_scores.extend(inner)
            chosen=select_loss(pd.DataFrame(inner))
            choices.append({"fold":fold,"selected_policy":chosen})
            schema=schema_for(tr)
            cols=schema.numeric+schema.categorical
            if cols!=archive_features[fold] or not ref.iloc[te].fold.eq(fold).all():
                raise ValueError("Archived control schema or TEST differs")
            part["features"]=cols
            partitions.append(part)
            values={BASE:ref.iloc[te].prediction.to_numpy(float)}
            for policy in LOSSES:
                if policy!=BASE:
                    values[policy]=fit(tr,te,policy,fold,"outer",schema)
            values[SELECTED]=values[chosen]
            for policy,pred in values.items():
                m=score(y.iloc[te],pred)
                metrics.append({"fold":fold,"policy":policy,**m})
                predictions.extend({"fold":fold,"policy":policy,"row_id":str(x.index[i]),
                    "group":str(groups.iloc[i]),"truth":float(y.iloc[i]),"prediction":float(v)}
                    for i,v in zip(te,pred,strict=True))
                print(f"Fold {fold}/5 | {policy} | RMSE={m['rmse']:.4f} | R2={m['r2']:.4f}",flush=True)
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
                raise ValueError("OOF identities differ")
            fm=pd.DataFrame([score(g.truth,g.prediction) for _,g in p.groupby("fold")])
            agg=summarize_scores(fm)
            tail=p.loc[p.truth.gt(TAIL)]
            global_m=regression_metrics(p.truth,p.prediction)
            report.append({"policy":policy,**agg,"rmse_global":global_m.rmse,"r2_global":global_m.r2,
                "rmse_std":float(fm.rmse.std(ddof=1)),"tail_bias":float((tail.prediction-tail.truth).mean()),
                **benchmark_status(agg["rmse_mean"],agg["r2_mean"])})
            for band,g in p.groupby(pd.cut(p.truth,BINS),observed=True):
                bands.append({"policy":policy,"band":str(band),"rows":len(g),
                              **regression_metrics(g.truth,g.prediction).to_dict()})
        base=next(r for r in report if r["policy"]==BASE)
        if fits!=MAX_FITS or len(report)!=len(LOSSES)+1 or file_sha256(source)!=digest:
            raise ValueError("Training loss comparison incomplete")
        for r in report:
            r["development_improvement"]=bool(r["rmse_mean"]<base["rmse_mean"]
                and r["tail_rmse"]<=base["tail_rmse"]
                and r["mae_mean"]<=base["mae_mean"]+MAE_TOLERANCE and r["r2_mean"]>.499)
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
