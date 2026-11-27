"""Nested learned stacking reusing archived HGB crossfit predictions."""
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
from lithiumscope.datasets.management.sources.local import local_training_source
from lithiumscope.experiments.model_1.protocols.cohorts import cohort
from lithiumscope.experiments.model_1.studies.expansion.academic_bridge import PILOT,SEED,check_pilot,benchmark_status
from lithiumscope.experiments.model_1.studies.expansion.academic_final import load_reference,FEATURE_SHA
from lithiumscope.experiments.model_1.studies.expansion.academic_features import load_recovery,measurement_blocks,select_columns,RECOVERY_SHA
from lithiumscope.experiments.model_1.training_methods.boosting import archived_run,archive_predictions,HGB_RUN,HGB_SHA,check_partition,fit_method,PROFILES_NEW
from lithiumscope.experiments.model_1.training_methods.losses import fit_loss,score,summarize_scores,BINS,TAIL,MAE_TOLERANCE
from lithiumscope.experiments.model_1.training_methods.stacking import BASE,POLICIES,fit_stack
from lithiumscope.experiments.model_1.studies.integration.posttraining import crossfit
from lithiumscope.experiments.model_1.data_expansion.sensitivity_data import read
from lithiumscope.model_1.schema import validate_training_schema
from lithiumscope.model_1.training.preprocessing.tabular import FeatureSchema
from lithiumscope.resources.detection.device import detect_device
from lithiumscope.results.atomic_exports import write_table,write_legacy_evidence_json

MAX_BASE_FITS=60
MAX_CALIBRATION_SETS=20
POST_RUN="20261126T223705_0c698520"
POST_SHA="e3c94c67492c219925b8e9ccc355d9e75d2dcd99c4f677f94ce38c95da31c852"
BOOST_RUN="20261126T221622_b1d7f8b8"
BOOST_SHA="ba2090c1e92a62a669fe175382db61d8e124baf2cd7f38eee0977c2d988e7f9f"
XGB="xgb_regularized"
PRIOR="prior_calibration_selected"
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


def archived_oof(table,fold,stage,x,y,pos):
    part=table.loc[table.fold.eq(fold)&table.stage.eq(stage)].set_index("row_id")
    ids=x.index[pos].astype(str)
    if not part.index.is_unique or set(part.index)!=set(ids):
        raise ValueError("Archived HGB crossfit identities differ")
    part=part.reindex(ids)
    p=part.oof_prediction.to_numpy(float)
    if not np.allclose(part.truth,y.iloc[pos],rtol=0,atol=1e-8) or not np.isfinite(p).all() or (p<0).any():
        raise ValueError("Archived HGB crossfit values differ")
    return p


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
    pr,pm=archived_run("model_1_integral_posttraining",POST_RUN,POST_SHA,digest)
    br,bm=archived_run("model_1_integral_boosting_methods",BOOST_RUN,BOOST_SHA,digest)
    if pm["hgb_manifest_sha256"]!=HGB_SHA or bm["hgb_manifest_sha256"]!=HGB_SHA:
        raise ValueError("Archived HGB dependency differs")
    if bm["new_profiles"][XGB]!=list(PROFILES_NEW[XGB]):
        raise ValueError("Archived XGB profile differs")
    xgb=archive_predictions(br,XGB,pilot,x)
    prior=archive_predictions(pr,"inner_selected",pilot,x)
    po=archive_predictions(pr,"identity",pilot,x)
    if not np.allclose(po.prediction,hgb.prediction,rtol=0,atol=1e-10):
        raise ValueError("Archived posttraining base differs")
    co=pd.read_csv(pr/"calibration_oof.csv",dtype={"row_id":str})
    pft={(v["fold"],v["stage"]):v for v in read(pr/"fit_trace.json")}
    pct={(v["fold"],v["stage"]):v for v in read(pr/"calibration_trace.json")}
    bx=pd.read_csv(br/"inner_scores.csv")
    bx=bx.loc[bx.policy.eq(XGB)]
    btrace={(v["fold"],v["stage"]):v for v in read(br/"fit_trace.json") if v["policy"]==XGB}
    if len(co)!=39156 or len(pft)!=60 or len(pct)!=20 or len(bx)!=15 or len(btrace)!=20:
        raise ValueError("Archived stacking evidence incomplete")
    recovered,rm=load_recovery(prepared,digest)
    if not recovered.academic_member.eq(keep).all():
        raise ValueError("Recovery membership differs")
    measured,ratios=measurement_blocks(frame,recovered.loc[keep,rm["measurement_columns"]],x)
    features=pd.concat([x,measured,ratios],axis=1)
    numeric,categorical=list(prepared.schema.numeric),list(prepared.schema.categorical)
    validate_training_schema(prepared.target,numeric+list(measured)+list(ratios),categorical)
    x=x[numeric+categorical]
    out=RESULTS_DIR/"generalization/model_1_integral_stacking"/(
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")+"_"+uuid4().hex[:8])
    out.mkdir(parents=True,exist_ok=False)
    manifest={"state":"running","scope":"nested_learned_stacking_academic_development",
        "runtime":runtime_fingerprint(),"source_sha256":digest,"feature_manifest_sha256":FEATURE_SHA,
        "recovery_manifest_sha256":RECOVERY_SHA,"hgb_manifest_sha256":HGB_SHA,"rows":len(x),"domain":domain,
        "posttraining_manifest_sha256":POST_SHA,"boosting_manifest_sha256":BOOST_SHA,
        "xgb_profile":PROFILES_NEW[XGB][1],"policies":POLICIES,"descriptive_controls":[XGB,PRIOR],"max_base_fits":MAX_BASE_FITS,"max_calibration_sets":MAX_CALIBRATION_SETS,
        "archived_outer_base":True,"archived_hgb_crossfit":True,"outer_folds":5,"inner_folds":3,"crossfit_folds":3,"seed":SEED,
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

    def check_trace(tr,te,fold,stage,schema):
        row=pft[fold,stage]
        if row["train_ids"]!=ids(tr) or row["validation_ids"]!=ids(te) or row["features"]!=schema.numeric+schema.categorical:
            raise ValueError("Archived HGB crossfit partition/schema differs")

    def base_fit(tr,te,fold,stage):
        nonlocal fits
        if np.intersect1d(tr,te).size:
            raise ValueError("Base fit overlap")
        schema=schema_for(tr)
        cols=schema.numeric+schema.categorical
        check_trace(tr,te,fold,stage,schema)
        fits+=1
        p=fit_method(features.iloc[tr][cols],y.iloc[tr],features.iloc[te][cols],schema,device,XGB)
        trace.append({"fold":fold,"stage":stage,"train_ids":ids(tr),"validation_ids":ids(te),"features":cols})
        return p

    def calibrate(tr,te,hc,xc,hp,xp,fold,stage):
        if np.intersect1d(tr,te).size:
            raise ValueError("Stacking TRAIN/evaluation overlap")
        old=pct[fold,stage]
        if old["calibration_ids"]!=ids(tr) or old["evaluation_ids"]!=ids(te):
            raise ValueError("Archived calibration partition differs")
        predictor,info=fit_stack(x.iloc[tr],y.iloc[tr],pd.Series(hc,index=x.index[tr]),
            pd.Series(xc,index=x.index[tr]),numeric,categorical,device)
        cal_trace.append({"fold":fold,"stage":stage,"calibration_ids":ids(tr),"evaluation_ids":ids(te),**info})
        cal_rows.extend({"fold":fold,"stage":stage,"row_id":str(x.index[i]),
                         "truth":float(y.iloc[i]),"hgb_oof":float(h),"xgb_oof":float(v)}
                        for i,h,v in zip(tr,hc,xc,strict=True))
        return predictor(x.iloc[te],hp,xp)

    try:
        for fold,(tr,te) in enumerate(splits,1):
            schema=schema_for(tr)
            cols=schema.numeric+schema.categorical
            part=old_parts[fold-1]
            check_partition(part,fold,x,tr,te,cols)
            if cols!=archive_cols[fold] or cols!=btrace[fold,"outer"]["features"]:
                raise ValueError("Archived outer features differ")
            inner=[]
            outer_cal=np.full(len(tr),np.nan)
            hc_outer=archived_oof(co,fold,"outer",x,y,tr)
            for j,(a,b) in enumerate(KFold(3,shuffle=True,random_state=SEED+fold).split(tr),1):
                it,iv=tr[a],tr[b]
                sub=part["inner"][j-1]
                if sub["inner_fold"]!=j or sub["train_ids"]!=ids(it) or sub["validation_ids"]!=ids(iv):
                    raise ValueError("Archived inner partition differs")
                hc=archived_oof(co,fold,f"inner_{j}",x,y,it)
                xc,_=crossfit(it,lambda u,v,k:base_fit(u,v,fold,f"inner_{j}_crossfit_{k}"),
                              SEED+100*fold+j)
                xp=base_fit(it,iv,fold,f"inner_{j}_base")
                hp=hc_outer[b]
                outer_cal[b]=xp
                for pred,archive in ((hp,old_inner),(xp,bx)):
                    mm=score(y.iloc[iv],pred)
                    prev=archive.loc[archive.fold.eq(fold)&archive.inner_fold.eq(j)].iloc[0]
                    if not np.allclose([mm[k] for k in ("rmse","mae","r2","tail_sse")],
                                       [prev[k] for k in ("rmse","mae","r2","tail_sse")],rtol=1e-6,atol=1e-6):
                        raise ValueError("Reconstructed inner base differs")
                if schema_for(it).numeric+categorical!=btrace[fold,f"inner_{j}"]["features"]:
                    raise ValueError("Archived XGB inner schema differs")
                values=calibrate(it,iv,hc,xc,hp,xp,fold,f"inner_{j}")
                for policy,p in values.items():
                    inner.append({"fold":fold,"inner_fold":j,"policy":policy,**score(y.iloc[iv],p)})
                print(f"Outer {fold}/5 | stacking inner {j}/3",flush=True)
            inner_scores.extend(inner)
            if not np.isfinite(outer_cal).all():
                raise ValueError("Incomplete outer TRAIN stacking OOF")
            chosen=choose(pd.DataFrame(inner))
            choices.append({"fold":fold,"selected_policy":chosen})
            if any(not a.iloc[te].fold.eq(fold).all() for a in (hgb,xgb,prior)):
                raise ValueError("Outer control differs")
            xp=xgb.iloc[te].prediction.to_numpy(float)
            values=calibrate(tr,te,hc_outer,outer_cal,hgb.iloc[te].prediction.to_numpy(float),xp,fold,"outer")
            values[SELECTED]=values[chosen]
            values[XGB]=xp
            values[PRIOR]=prior.iloc[te].prediction.to_numpy(float)
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
        if fits!=MAX_BASE_FITS or len(cal_trace)!=MAX_CALIBRATION_SETS or len(report)!=len(POLICIES)+3 or file_sha256(source)!=digest:
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
        manifest["meta_estimators"]=sum(t["meta_estimators"] for t in cal_trace)
        manifest["convex_weight_fits"]=sum(t["convex_weight_fits"] for t in cal_trace)
        manifest["outputs"]={p.name:file_sha256(p) for p in out.iterdir() if p.is_file() and p.name!="manifest.json"}
        write_legacy_evidence_json(out/"manifest.json",manifest)
    print(pd.DataFrame(report).to_string(index=False))
    print(f"Estado: {manifest['state']} | Evidencias: {out}",flush=True)
    return out


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-unverified-metadata",action="store_true")
    run(allow_unverified_metadata=parser.parse_args().allow_unverified_metadata)
