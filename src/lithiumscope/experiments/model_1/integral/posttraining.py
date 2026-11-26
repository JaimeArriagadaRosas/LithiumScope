"""Nested cross-fitted point calibration of the frozen HGB academic control."""
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
from lithiumscope.experiments.model_1.expansion.academic_bridge import PILOT,SEED,check_pilot,benchmark_status
from lithiumscope.experiments.model_1.expansion.academic_final import load_reference,FEATURE_SHA
from lithiumscope.experiments.model_1.expansion.academic_features import load_recovery,measurement_blocks,select_columns,RECOVERY_SHA
from lithiumscope.experiments.model_1.integral.boosting_methods import archived_run,archive_predictions,HGB_RUN,HGB_SHA,check_partition
from lithiumscope.experiments.model_1.integral.training_losses import fit_loss,score,summarize_scores,BINS,TAIL,MAE_TOLERANCE
from lithiumscope.experiments.model_1.integral.calibration import BASE,POLICIES,fit_calibrators
from lithiumscope.experiments.model_1.expansion.sensitivity_data import read
from lithiumscope.model_1.schema import validate_training_schema
from lithiumscope.model_1.training.preprocessing.tabular import FeatureSchema
from lithiumscope.resources.detection.device import detect_device
from lithiumscope.results.atomic_exports import write_table,write_legacy_evidence_json

MAX_BASE_FITS=60
MAX_CALIBRATION_SETS=20
SELECTED="inner_selected"


def choose(frame):
    if (set(frame.policy)!=set(POLICIES) or frame.duplicated(["policy","inner_fold"]).any()
            or any(set(g.inner_fold)!={1,2,3} for _,g in frame.groupby("policy"))):
        raise ValueError("Incomplete inner calibration evidence")
    stats={p:summarize_scores(g) for p,g in frame.groupby("policy")}
    control=stats[BASE]
    eligible=[p for p,v in stats.items() if p!=BASE and v["rmse_mean"]<control["rmse_mean"]
        and v["r2_mean"]>.499 and v["mae_mean"]<=control["mae_mean"]+MAE_TOLERANCE
        and v["tail_rmse"]<=control["tail_rmse"]]
    return min(eligible,key=lambda p:(stats[p]["rmse_mean"],stats[p]["mae_mean"],p)) if eligible else BASE


def crossfit(tr,fit,seed):
    """Return one base prediction per TRAIN identity; fit callback sees only disjoint positions."""
    tr=np.asarray(tr,dtype=int)
    if tr.ndim!=1 or len(np.unique(tr))!=len(tr) or len(tr)<3:
        raise ValueError("Invalid crossfit population")
    result=np.full(len(tr),np.nan)
    parts=[]
    for j,(a,b) in enumerate(KFold(3,shuffle=True,random_state=seed).split(tr),1):
        pred=np.asarray(fit(tr[a],tr[b],j),float)
        if pred.shape!=(len(b),) or not np.isfinite(pred).all() or (pred<0).any():
            raise ValueError("Invalid crossfit predictions")
        result[b]=pred
        parts.append((tr[a],tr[b]))
    if not np.isfinite(result).all():
        raise ValueError("Incomplete crossfit OOF")
    return result,parts


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
    reference,archive_cols=load_reference(x,pilot,digest)
    hr,_=archived_run("model_1_integral_training_losses",HGB_RUN,HGB_SHA,digest)
    hgb=archive_predictions(hr,"squared_error",pilot,x)
    if not np.allclose(hgb.prediction,reference.prediction,rtol=0,atol=1e-10):
        raise ValueError("Archived HGB control differs")
    old_parts=read(hr/"partitions.json")
    old_inner=pd.read_csv(hr/"inner_scores.csv")
    old_inner=old_inner.loc[old_inner.policy.eq("squared_error")]
    recovered,rm=load_recovery(prepared,digest)
    if not recovered.academic_member.eq(keep).all():
        raise ValueError("Recovery membership differs")
    measured,ratios=measurement_blocks(frame,recovered.loc[keep,rm["measurement_columns"]],x)
    features=pd.concat([x,measured,ratios],axis=1)
    numeric,categorical=list(prepared.schema.numeric),list(prepared.schema.categorical)
    validate_training_schema(prepared.target,numeric+list(measured)+list(ratios),categorical)
    x=x[numeric+categorical]
    out=RESULTS_DIR/"generalization/model_1_integral_posttraining"/(
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")+"_"+uuid4().hex[:8])
    out.mkdir(parents=True,exist_ok=False)
    manifest={"state":"running","scope":"nested_crossfit_point_calibration_academic_development",
        "runtime":runtime_fingerprint(),"source_sha256":digest,"feature_manifest_sha256":FEATURE_SHA,
        "recovery_manifest_sha256":RECOVERY_SHA,"hgb_manifest_sha256":HGB_SHA,"rows":len(x),"domain":domain,
        "policies":POLICIES,"max_base_fits":MAX_BASE_FITS,"max_calibration_sets":MAX_CALIBRATION_SETS,
        "archived_outer_base":True,"outer_folds":5,"inner_folds":3,"crossfit_folds":3,"seed":SEED,
        "random_cv_spatial_overlap":True,"reused_development_folds":True,
        "retrospective_target_assisted_recovery":True,"metadata_certified":False,
        "independent_holdout":False,"direct_comparability":False,"new_training_rows":0,
        "new_predictors":False,"production_schema_changed":False,"models_saved":False,
        "release_eligible":False,"promotion":False}
    write_legacy_evidence_json(out/"manifest.json",manifest)
    device=detect_device(prefer_gpu=False)
    fits=0
    predictions,metrics,inner_scores,trace,cal_trace,choices,cal_rows=[],[],[],[],[],[],[]

    def ids(pos):
        return x.index[pos].astype(str).tolist()

    def schema_for(tr):
        extra=select_columns(measured.iloc[tr],groups.iloc[tr])+select_columns(ratios.iloc[tr],groups.iloc[tr])
        return FeatureSchema(numeric+extra,categorical,2,"stable_v1")

    def base_fit(tr,te,fold,stage):
        nonlocal fits
        if np.intersect1d(tr,te).size:
            raise ValueError("Base fit overlap")
        schema=schema_for(tr)
        cols=schema.numeric+schema.categorical
        fits+=1
        p=fit_loss(features.iloc[tr][cols],y.iloc[tr],features.iloc[te][cols],schema,device,"squared_error")
        trace.append({"fold":fold,"stage":stage,"train_ids":ids(tr),"validation_ids":ids(te),"features":cols})
        return p

    def calibrate(tr,te,calp,pred,fold,stage):
        if np.intersect1d(tr,te).size:
            raise ValueError("Calibration TRAIN/evaluation overlap")
        predictor,info=fit_calibrators(x.iloc[tr],y.iloc[tr],pd.Series(calp,index=x.index[tr]),numeric,categorical)
        cal_trace.append({"fold":fold,"stage":stage,"calibration_ids":ids(tr),"evaluation_ids":ids(te),**info})
        cal_rows.extend({"fold":fold,"stage":stage,"row_id":str(x.index[i]),
                         "truth":float(y.iloc[i]),"oof_prediction":float(v)}
                        for i,v in zip(tr,calp,strict=True))
        return predictor(x.iloc[te],pred)

    try:
        for fold,(tr,te) in enumerate(splits,1):
            schema=schema_for(tr)
            cols=schema.numeric+schema.categorical
            part=old_parts[fold-1]
            check_partition(part,fold,x,tr,te,cols)
            if cols!=archive_cols[fold]:
                raise ValueError("Archived outer features differ")
            inner=[]
            outer_cal=np.full(len(tr),np.nan)
            for j,(a,b) in enumerate(KFold(3,shuffle=True,random_state=SEED+fold).split(tr),1):
                it,iv=tr[a],tr[b]
                sub=part["inner"][j-1]
                if sub["inner_fold"]!=j or sub["train_ids"]!=ids(it) or sub["validation_ids"]!=ids(iv):
                    raise ValueError("Archived inner partition differs")
                # Independent crossfit within this inner TRAIN; IV is never used to fit a correction.
                cp,_=crossfit(it,lambda u,v,k:base_fit(u,v,fold,f"inner_{j}_crossfit_{k}"),
                              SEED+100*fold+j)
                ip=base_fit(it,iv,fold,f"inner_{j}_base")
                # Retain these untouched inner-validation predictions for outer TRAIN calibration.
                outer_cal[b]=ip
                mm=score(y.iloc[iv],ip)
                prev=old_inner.loc[old_inner.fold.eq(fold)&old_inner.inner_fold.eq(j)].iloc[0]
                if not np.allclose([mm[k] for k in ("rmse","mae","r2","tail_sse")],
                                   [prev[k] for k in ("rmse","mae","r2","tail_sse")],rtol=1e-6,atol=1e-6):
                    raise ValueError("Reconstructed inner HGB control differs")
                values=calibrate(it,iv,cp,ip,fold,f"inner_{j}")
                for policy,p in values.items():
                    inner.append({"fold":fold,"inner_fold":j,"policy":policy,**score(y.iloc[iv],p)})
                print(f"Outer {fold}/5 | calibration inner {j}/3",flush=True)
            inner_scores.extend(inner)
            if not np.isfinite(outer_cal).all():
                raise ValueError("Incomplete outer TRAIN calibration OOF")
            chosen=choose(pd.DataFrame(inner))
            choices.append({"fold":fold,"selected_policy":chosen})
            if not hgb.iloc[te].fold.eq(fold).all():
                raise ValueError("Outer HGB control differs")
            values=calibrate(tr,te,outer_cal,hgb.iloc[te].prediction.to_numpy(float),fold,"outer")
            values[SELECTED]=values[chosen]
            for policy,p in values.items():
                mm=score(y.iloc[te],p)
                metrics.append({"fold":fold,"policy":policy,**mm})
                predictions.extend({"fold":fold,"policy":policy,"row_id":str(x.index[i]),"group":str(groups.iloc[i]),
                    "truth":float(y.iloc[i]),"prediction":float(v)} for i,v in zip(te,p,strict=True))
                print(f"Fold {fold}/5 | {policy} | RMSE={mm['rmse']:.4f} | R2={mm['r2']:.4f}",flush=True)
            write_table(out/"predictions.csv",pd.DataFrame(predictions))
            write_table(out/"fold_metrics.csv",pd.DataFrame(metrics))
            write_table(out/"inner_scores.csv",pd.DataFrame(inner_scores))
            write_table(out/"calibration_oof.csv",pd.DataFrame(cal_rows))
            write_legacy_evidence_json(out/"fit_trace.json",trace)
            write_legacy_evidence_json(out/"calibration_trace.json",cal_trace)
            write_legacy_evidence_json(out/"selection.json",choices)
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
        if fits!=MAX_BASE_FITS or len(cal_trace)!=MAX_CALIBRATION_SETS or len(report)!=len(POLICIES)+1 or file_sha256(source)!=digest:
            raise ValueError("Posttraining comparison incomplete")
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
        manifest["base_fits"]=fits
        manifest["calibration_sets"]=len(cal_trace)
        manifest["calibration_estimators"]=sum(t["fitted_estimators"] for t in cal_trace)
        manifest["calibration_bias_statistics"]=sum(t["fitted_bias_statistics"] for t in cal_trace)
        manifest["outputs"]={p.name:file_sha256(p) for p in out.iterdir() if p.is_file() and p.name!="manifest.json"}
        write_legacy_evidence_json(out/"manifest.json",manifest)
    print(pd.DataFrame(report).to_string(index=False))
    print(f"Estado: {manifest['state']} | Evidencias: {out}",flush=True)
    return out


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-unverified-metadata",action="store_true")
    run(allow_unverified_metadata=parser.parse_args().allow_unverified_metadata)
