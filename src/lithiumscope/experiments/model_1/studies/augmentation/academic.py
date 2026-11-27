"""Compare enriched TRAIN treatments on the unchanged academic development folds."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from uuid import uuid4
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.paths import RESULTS_DIR
from lithiumscope.core.scientific.eda_statistics import source_series
from lithiumscope.core.scientific.regression_metrics import regression_metrics
from lithiumscope.datasets.management.sources.local import local_training_source
from lithiumscope.experiments.model_1.protocols.cohorts import cohort
from lithiumscope.experiments.model_1.studies.expansion.academic_bridge import PILOT, SEED, PROFILES, check_pilot, benchmark_status
from lithiumscope.experiments.model_1.studies.expansion.academic_final import load_reference, FEATURE_SHA
from lithiumscope.experiments.model_1.studies.expansion.academic_features import RECOVERY_SHA, load_recovery, measurement_blocks, select_columns
from lithiumscope.experiments.model_1.treatments.augmentation import ARMS, TAIL, augment
from lithiumscope.model_1.schema import validate_training_schema
from lithiumscope.model_1.training.factory import build_pipeline
from lithiumscope.model_1.training.preprocessing.tabular import FeatureSchema
from lithiumscope.resources.detection.device import detect_device
from lithiumscope.resources.manager import estimator_thread_scope
from lithiumscope.results.atomic_exports import write_table, write_legacy_evidence_json
from lithiumscope.results.reference_benchmarks import OASIS_M1

BASE="control"
SELECTED="inner_selected"


def choose(scores):
    if (set(scores.policy)!=set(ARMS) or scores.duplicated(["policy","inner_fold"]).any()
            or any(set(v.inner_fold)!={1,2,3} for _,v in scores.groupby("policy"))
            or not np.isfinite(scores[["rmse","r2"]]).all().all()):
        raise ValueError("Incomplete inner selection")
    agg=scores.groupby("policy")[["rmse","r2"]].mean().reset_index()
    control=agg.loc[agg.policy.eq(BASE)].iloc[0]
    feasible=agg.loc[agg.r2.gt(OASIS_M1["r2_mean"])&agg.rmse.lt(control.rmse-1e-10)]
    return BASE if feasible.empty else str(feasible.sort_values(["rmse","r2","policy"],
        ascending=[True,False,True]).iloc[0].policy)


def run(*,allow_unverified_metadata=False):
    if not allow_unverified_metadata:
        raise ValueError("Explicit --allow-unverified-metadata required; research only")
    source=local_training_source("model_1")
    digest=file_sha256(source)
    prepared,keep,domain=cohort(source)
    x,y,groups,frame=(v.loc[keep] for v in (prepared.x,prepared.y,prepared.groups,prepared.frame))
    if len(x)!=3263 or len(prepared.x)!=3420 or len(x.columns)!=33:
        raise ValueError("Frozen academic cohort differs")
    sources=source_series(frame)
    if sources.eq("unknown").any():
        raise ValueError("Missing historical source identity")
    splits=list(KFold(5,shuffle=True,random_state=SEED).split(x))
    pilot,_=check_pilot(RESULTS_DIR/"generalization/model_1_oasis_like"/PILOT,x,y,groups,splits,digest,domain)
    ref,archived_features=load_reference(x,pilot,digest)
    r,m=load_recovery(prepared,digest)
    if not r.academic_member.eq(keep).all():
        raise ValueError("Recovery membership differs")
    measured,ratios=measurement_blocks(frame,r.loc[keep,m["measurement_columns"]],x)
    numeric,categorical=list(prepared.schema.numeric),list(prepared.schema.categorical)
    validate_training_schema(prepared.target,numeric+list(measured)+list(ratios),categorical)
    features=pd.concat([x,measured,ratios],axis=1)
    out=RESULTS_DIR/"generalization/model_1_enriched_augmentation"/(
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")+"_"+uuid4().hex[:8])
    out.mkdir(parents=True,exist_ok=False)
    manifest={"state":"running","scope":"enriched_augmentation_academic_development",
        "source_sha256":digest,"feature_manifest_sha256":FEATURE_SHA,"recovery_manifest_sha256":RECOVERY_SHA,
        "domain":domain,"rows":len(x),"arms":ARMS,"seed":SEED,"outer_folds":5,"inner_folds":3,
        "max_new_fits":115,"tail_threshold_ppm":TAIL,"budget_per_train_tail_row":2,
        "oasis_reference":OASIS_M1,"archived_outer_control_reused":True,
        "retrospective_target_assisted_record_linkage":True,"random_cv_spatial_overlap":True,
        "direct_comparability":False,"independent_holdout":False,"metadata_certified":False,
        "new_real_rows":0,"synthetic_evaluation_rows":0,"release_eligible":False,
        "promotion":False,"models_saved":False,"production_schema_changed":False}
    write_legacy_evidence_json(out/"manifest.json",manifest)
    device=detect_device(prefer_gpu=False)
    params={k:v for k,v in PROFILES["hist_gradient_boosting"].items() if k!="treatment_policy"}
    params["min_samples_leaf"]=5
    fits=0
    pred_rows,inner_rows,score_rows,metrics,traces,lineage,choices,partitions=[],[],[],[],[],[],[],[]

    def partition(tr,te,fold,stage):
        nonlocal fits
        if np.intersect1d(tr,te).size:
            raise ValueError("Evaluation overlaps TRAIN")
        extra=select_columns(measured.iloc[tr],groups.iloc[tr])+select_columns(ratios.iloc[tr],groups.iloc[tr])
        schema=FeatureSchema(numeric+extra,categorical,2,"stable_v1")
        cols=schema.numeric+schema.categorical
        # Generate from raw measurements only, including parents needed to recompute
        # admitted ratios; the estimator sees only the TRAIN-admitted columns.
        tx=features.iloc[tr]
        vx=features.iloc[te][cols]
        partitions.append({"fold":fold,"stage":stage,"train_ids":x.index[tr].astype(str).tolist(),
                           "evaluation_ids":x.index[te].astype(str).tolist(),"features":cols})
        values={}
        for policy in ARMS:
            seed=SEED+100*fold+(0 if stage=="outer" else int(stage.split("_")[-1]))
            xx,yy,w,parents,info=augment(tx,y.iloc[tr],groups.iloc[tr],sources.iloc[tr],
                                        list(x.columns),list(measured),categorical,policy,seed)
            allowed=set(x.index[tr].astype(str))
            if any(str(row[k]) not in allowed for row in parents for k in ("parent_a","parent_b") if k in row):
                raise ValueError("Synthetic parent outside TRAIN")
            if len(w)!=len(yy) or not np.isfinite(w).all() or (w<=0).any():
                raise ValueError("Invalid fit weights")
            reused=False
            if stage=="outer" and policy==BASE:
                if cols!=archived_features[fold] or not ref.iloc[te].fold.eq(fold).all():
                    raise ValueError("Archived control schema differs")
                score=ref.iloc[te].prediction.to_numpy(float)
                reused=True
            elif policy in ("rare_copies","relevance_copies","smoter_enriched","gaussian_enriched") and not parents:
                score=values[BASE].copy()
                reused=True
            else:
                model=build_pipeline("hist_gradient_boosting",device,SEED,schema,params)
                model.set_params(preprocess__numeric="passthrough")
                with estimator_thread_scope():
                    model.fit(xx[cols],yy,model__sample_weight=w)
                fits+=1
                score=np.maximum(np.asarray(model.predict(vx),float),0.)
            if score.shape!=(len(te),) or not np.isfinite(score).all() or (score<0).any():
                raise ValueError("Invalid evaluation predictions")
            values[policy]=score
            traces.append({"fold":fold,"stage":stage,"policy":policy,"seed":seed,"control_reused":reused,
                           "weight_min":float(w.min()),"weight_max":float(w.max()),**info})
            lineage.extend({"fold":fold,"stage":stage,"policy":policy,"fit_position":len(tx)+i,**row}
                           for i,row in enumerate(parents))
        return values

    def checkpoint():
        for name,rows in (("predictions.csv",pred_rows),("inner_predictions.csv",inner_rows),
                          ("inner_scores.csv",score_rows),("fold_metrics.csv",metrics),
                          ("fit_trace.csv",traces)):
            write_table(out/name,pd.DataFrame(rows))
        write_legacy_evidence_json(out/"selection.json",choices)
        write_legacy_evidence_json(out/"lineage.json",lineage)
        write_legacy_evidence_json(out/"partitions.json",partitions)

    try:
        for fold,(tr,te) in enumerate(splits,1):
            scores=[]
            for number,(a,b) in enumerate(KFold(3,shuffle=True,random_state=SEED+fold).split(tr),1):
                values=partition(tr[a],tr[b],fold,f"inner_{number}")
                for policy,score in values.items():
                    scores.append({"fold":fold,"inner_fold":number,"policy":policy,
                                   **regression_metrics(y.iloc[tr[b]],score).to_dict()})
                    inner_rows.extend({"fold":fold,"inner_fold":number,"policy":policy,"row_id":str(x.index[i]),
                                       "truth":float(y.iloc[i]),"prediction":float(v)}
                                      for i,v in zip(tr[b],score,strict=True))
                print(f"Outer {fold}/5 | inner {number}/3 | {len(ARMS)} treatments",flush=True)
            score_rows.extend(scores)
            chosen=choose(pd.DataFrame(scores))
            choices.append({"fold":fold,"selected_policy":chosen})
            values=partition(tr,te,fold,"outer")
            values[SELECTED]=values[chosen]
            for policy,score in values.items():
                mm=regression_metrics(y.iloc[te],score).to_dict()
                metrics.append({"fold":fold,"policy":policy,**mm})
                pred_rows.extend({"fold":fold,"policy":policy,"row_id":str(x.index[i]),"group":str(groups.iloc[i]),
                                  "truth":float(y.iloc[i]),"prediction":float(v)}
                                 for i,v in zip(te,score,strict=True))
                print(f"Fold {fold}/5 | {policy} | RMSE={mm['rmse']:.4f} | R2={mm['r2']:.4f}",flush=True)
            checkpoint()
        report,bands=[],[]
        predictions=pd.DataFrame(pred_rows)
        for policy,part in predictions.groupby("policy"):
            part=part.sort_values("row_id").reset_index(drop=True)
            if (not part.row_id.is_unique or not part[["row_id","fold","group"]].equals(pilot[["row_id","fold","group"]])
                    or not np.allclose(part.truth,pilot.truth,rtol=0,atol=1e-8)):
                raise ValueError("Outer identities differ")
            fm=pd.DataFrame([regression_metrics(g.truth,g.prediction).to_dict() for _,g in part.groupby("fold")])
            rmse,r2=float(fm.rmse.mean()),float(fm.r2.mean())
            report.append({"policy":policy,"rows":len(part),"rmse_mean":rmse,"r2_mean":r2,
                           "mae_mean":float(fm.mae.mean()),**benchmark_status(rmse,r2)})
            part=part.assign(band=pd.cut(part.truth,[0,5,10,20,40,60,80,100,np.inf]))
            for band,g in part.groupby("band",observed=True):
                bands.append({"policy":policy,"band":str(band),"rows":len(g),
                              **regression_metrics(g.truth,g.prediction).to_dict()})
        if len(report)!=len(ARMS)+1 or len(traces)!=120 or len(partitions)!=20 or fits>115 or file_sha256(source)!=digest:
            raise ValueError("Campaign incomplete")
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
        manifest["evaluated_treatments"]=len(traces)
        manifest["outputs"]={p.name:file_sha256(p) for p in out.iterdir() if p.is_file() and p.name!="manifest.json"}
        write_legacy_evidence_json(out/"manifest.json",manifest)
    print(pd.DataFrame(report).to_string(index=False))
    print(pd.DataFrame(traces).query("stage == 'outer'")[["fold","policy","added_rows","compatible_pairs","fit_tail_rows"]].to_string(index=False))
    print(f"Estado: {manifest['state']} | Evidencias: {out}",flush=True)
    return out


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-unverified-metadata",action="store_true")
    run(allow_unverified_metadata=parser.parse_args().allow_unverified_metadata)
