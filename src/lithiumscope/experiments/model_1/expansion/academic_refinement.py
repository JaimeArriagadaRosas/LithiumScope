"""Nested academic HGB optimization on the exact previous academic TEST cohort."""
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
from lithiumscope.experiments.model_1.expansion.academic_bridge import PILOT, SEED, benchmark_status, check_pilot, range_pool
from lithiumscope.experiments.model_1.expansion.academic_treatments import BASE, PROFILES_HGB, fit_profile, select_profile
from lithiumscope.experiments.model_1.expansion.sensitivity_data import load_evidence, historical_tokens, reserve, fold_pools, read
from lithiumscope.model_1.training.preprocessing.tabular import FeatureSchema
from lithiumscope.resources.detection.device import detect_device
from lithiumscope.results.atomic_exports import write_table, write_legacy_evidence_json
from lithiumscope.results.reference_benchmarks import OASIS_M1

BRIDGE="20261126T153931_23ea0e09"
BRIDGE_SHA256="603d44eb4e49d948e16820f9ad7e08edf1cf1c8d00653030d57d4fdd91fc4ea4"


def load_bridge(folder, previous, digest, links):
    path=folder/"manifest.json"
    if file_sha256(path)!=BRIDGE_SHA256:
        raise ValueError("Previous academic bridge manifest differs")
    manifest=read(path)
    if (manifest.get("state")!="complete" or manifest.get("source_sha256")!=digest
            or any(manifest.get(k)!=v for k,v in links.items())
            or manifest.get("rows")!=3263 or manifest.get("attempted_fits")!=30):
        raise ValueError("Previous bridge evidence identity differs")
    pred_path=folder/"predictions.csv"
    if file_sha256(pred_path)!=manifest["outputs"]["predictions.csv"]:
        raise ValueError("Previous bridge predictions differ")
    rows=pd.read_csv(pred_path,dtype={"row_id":str})
    ref=rows.loc[rows.algorithm.eq("hist_gradient_boosting")&rows.policy.eq("historical")].copy()
    ref=ref.sort_values("row_id").reset_index(drop=True)
    if (not ref.row_id.is_unique or not ref[["row_id","fold","group"]].equals(previous[["row_id","fold","group"]])
            or not np.allclose(ref.truth,previous.truth,rtol=0,atol=1e-8)
            or not np.isfinite(ref.prediction).all()):
        raise ValueError("Previous HGB TEST identity differs")
    return ref


def run(run_path, *, allow_unverified_metadata=False):
    if not allow_unverified_metadata:
        raise ValueError("Explicit --allow-unverified-metadata required; research only")
    evidence,audit,features,pubs,links=load_evidence(run_path)
    source=local_training_source("model_1")
    digest=file_sha256(source)
    if digest!=evidence["source_sha256"]:
        raise ValueError("Historical source differs")
    prepared,keep,domain=cohort(source)
    if len(prepared.x)!=3420 or int(keep.sum())!=3263 or list(prepared.x.columns)!=evidence["features"]:
        raise ValueError("Academic historical cohort or schema differs")
    x,y,groups=(v.loc[keep] for v in (prepared.x,prepared.y,prepared.groups))
    numeric=list(prepared.schema.numeric)
    categorical=list(prepared.schema.categorical)
    if numeric!=evidence["numeric"] or categorical!=evidence["categorical"]:
        raise ValueError("Academic predictor schema differs")
    schema=FeatureSchema(numeric,categorical,2,"stable_v1")
    all_refs=historical_tokens(prepared.frame)
    refs=[all_refs[i] for i in np.flatnonzero(keep.to_numpy())]
    held,pool=reserve(audit,pubs,prepared.groups,all_refs)
    cx=features[list(x.columns)].copy()
    for key in numeric:
        cx[key]=pd.to_numeric(cx[key],errors="raise").astype(float)
    for key in categorical:
        cx[key]=cx[key].astype(object).where(cx[key].notna(),np.nan)
    cy=pd.to_numeric(features.Li_icpms,errors="raise")
    cg=features.spatial_group.astype(str)
    pool=range_pool(pool,cy,domain)&~audit.screening_state.eq("high_transfer_review").to_numpy()
    splits=list(KFold(5,shuffle=True,random_state=SEED).split(x))
    previous,_=check_pilot(RESULTS_DIR/"generalization/model_1_oasis_like"/PILOT,x,y,groups,splits,digest,domain)
    ref=load_bridge(RESULTS_DIR/"generalization/model_1_expansion_academic"/BRIDGE,previous,digest,links)
    device=detect_device(prefer_gpu=False)
    out=RESULTS_DIR/"generalization/model_1_expansion_academic_refinement"/(
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")+"_"+uuid4().hex[:8])
    out.mkdir(parents=True,exist_ok=False)
    manifest={**links,"state":"running","scope":"academic_nested_hgb_development_reused_outer_folds",
        "source_sha256":digest,"bridge_id":BRIDGE,"bridge_manifest_sha256":BRIDGE_SHA256,
        "domain":domain,"rows":len(x),"outer_seed":SEED,"outer_folds":5,"inner_folds":3,
        "profiles":PROFILES_HGB,"planned_fits":5*(3*len(PROFILES_HGB)+1),
        "oasis_reference":OASIS_M1,"direct_comparability":False,"random_cv_spatial_overlap":True,
        "metadata_certified":False,"independent_holdout":False,"release_eligible":False,
        "promotion":False,"models_saved":False,"reserved_rows_excluded":int(held.sum())}
    write_legacy_evidence_json(out/"manifest.json",manifest)
    predictions,metrics,inner_scores,inner_predictions,trace,plans,choices=[],[],[],[],[],[],[]
    attempted=0

    def partition(tr,blocked,fold,stage,need_candidates):
        if np.intersect1d(tr,blocked).size:
            raise ValueError("Historical TRAIN overlaps evaluation rows")
        blocked_refs=set().union(*(refs[i] for i in blocked))
        pos=(fold_pools(audit,cx,x.iloc[tr],numeric,pool,pubs,set(groups.iloc[blocked]),blocked_refs)["D4"]
             if need_candidates else np.array([],dtype=int))
        if need_candidates and not len(pos):
            raise ValueError("No compatible candidates for "+stage)
        if held[pos].any() or set(cg.iloc[pos])&set(groups.iloc[blocked]) or any(pubs[i]&blocked_refs for i in pos):
            raise ValueError("Candidate TRAIN overlaps excluded groups, references or reserve")
        plans.append({"fold":fold,"stage":stage,"historical_train_ids":x.index[tr].astype(str).tolist(),
            "blocked_historical_ids":x.index[blocked].astype(str).tolist(),
            "candidate_pool_ids":audit.candidate_id.iloc[pos].tolist()})
        return pos

    def fit(tr,pos,profile,fold,stage):
        nonlocal attempted
        selected=pos if PROFILES_HGB[profile][3]>0 else np.array([],dtype=int)
        bx,by=x.iloc[tr].copy(),y.iloc[tr].copy()
        if len(selected):
            ids=pd.Index("candidate:"+audit.candidate_id.iloc[selected].astype(str))
            bx=pd.concat([bx,cx.iloc[selected].set_axis(ids)])
            by=pd.concat([by,cy.iloc[selected].set_axis(ids)])
        mask=pd.Series(np.r_[np.zeros(len(tr),dtype=bool),np.ones(len(selected),dtype=bool)],index=bx.index)
        attempted+=1
        predict,info=fit_profile(bx,by,mask,schema,device,SEED,profile)
        trace.append({"fold":fold,"stage":stage,"profile":profile,**info})
        return predict

    def checkpoint():
        write_table(out/"predictions.csv",pd.DataFrame(predictions))
        write_table(out/"fold_metrics.csv",pd.DataFrame(metrics))
        write_table(out/"inner_scores.csv",pd.DataFrame(inner_scores))
        write_table(out/"inner_predictions.csv",pd.DataFrame(inner_predictions))
        write_legacy_evidence_json(out/"fit_trace.json",trace)
        write_legacy_evidence_json(out/"partitions.json",plans)
        write_legacy_evidence_json(out/"selection.json",choices)

    try:
        for fold,(tr,te) in enumerate(splits,1):
            scores=[]
            for number,(a,b) in enumerate(KFold(3,shuffle=True,random_state=SEED+fold).split(tr),1):
                a,b=tr[a],tr[b]
                stage=f"inner_{number}"
                pos=partition(a,np.sort(np.r_[te,b]),fold,stage,True)
                for profile in PROFILES_HGB:
                    predict=fit(a,pos,profile,fold,stage)
                    values=predict(x.iloc[b])
                    m=regression_metrics(y.iloc[b],values).to_dict()
                    scores.append({"fold":fold,"inner_fold":number,"profile":profile,**m})
                    inner_predictions.extend({"fold":fold,"inner_fold":number,"profile":profile,
                        "row_id":str(x.index[i]),"truth":float(y.iloc[i]),"prediction":float(v)}
                        for i,v in zip(b,values,strict=True))
                print(f"Outer {fold}/5 | inner {number}/3 | 7 profiles",flush=True)
            inner_scores.extend(scores)
            choice,detail=select_profile(pd.DataFrame(scores))
            choices.append({"fold":fold,"selected_profile":choice,"inner_means":detail.to_dict("records")})
            pos=partition(tr,te,fold,"outer",PROFILES_HGB[choice][3]>0)
            predict=fit(tr,pos,choice,fold,"outer")
            values=predict(x.iloc[te])
            m=regression_metrics(y.iloc[te],values).to_dict()
            metrics.append({"fold":fold,"profile":choice,**m})
            predictions.extend({"fold":fold,"profile":choice,"row_id":str(x.index[i]),
                "group":str(groups.iloc[i]),"truth":float(y.iloc[i]),"prediction":float(v)}
                for i,v in zip(te,values,strict=True))
            print(f"Outer {fold}/5 | selected={choice} | RMSE={m['rmse']:.4f} | R2={m['r2']:.4f}",flush=True)
            checkpoint()
        pred=pd.DataFrame(predictions).sort_values("row_id").reset_index(drop=True)
        if (not pred.row_id.is_unique or not pred[["row_id","fold","group"]].equals(ref[["row_id","fold","group"]])
                or not np.allclose(pred.truth,ref.truth,rtol=0,atol=1e-8)
                or attempted!=manifest["planned_fits"] or file_sha256(source)!=digest):
            raise ValueError("Academic OOF identity or fit budget differs")
        rows=[]
        for label,table in (("HGB_previous",ref),("HGB_inner_selected",pred)):
            fm=pd.DataFrame([regression_metrics(g.truth,g.prediction).to_dict() for _,g in table.groupby("fold")])
            rmse,r2=float(fm.rmse.mean()),float(fm.r2.mean())
            rows.append({"policy":label,"rows":len(table),"rmse_mean":rmse,
                "rmse_std":float(fm.rmse.std(ddof=1)),"r2_mean":r2,"r2_std":float(fm.r2.std(ddof=1)),
                "mae_mean":float(fm.mae.mean()),**{"global_"+k:v for k,v in regression_metrics(table.truth,table.prediction).to_dict().items()},
                **benchmark_status(rmse,r2)})
        bands=[]
        for label,table in (("HGB_previous",ref),("HGB_inner_selected",pred)):
            table=table.assign(band=pd.cut(table.truth,[0,5,10,20,40,60,80,100,np.inf]))
            for band,g in table.groupby("band",observed=True):
                bands.append({"policy":label,"band":str(band),"rows":len(g),
                    **regression_metrics(g.truth,g.prediction).to_dict()})
        write_table(out/"summary.csv",pd.DataFrame(rows))
        write_table(out/"band_metrics.csv",pd.DataFrame(bands))
        manifest["summary"]=rows
        manifest["state"]="complete"
    except BaseException as exc:
        manifest["state"]="interrupted"
        manifest["error"]=str(exc)
        raise
    finally:
        manifest["attempted_fits"]=attempted
        manifest["outputs"]={p.name:file_sha256(p) for p in out.iterdir() if p.is_file() and p.name!="manifest.json"}
        write_legacy_evidence_json(out/"manifest.json",manifest)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"Estado: {manifest['state']} | Evidencias: {out}",flush=True)
    return out


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",required=True)
    parser.add_argument("--allow-unverified-metadata",action="store_true")
    args=parser.parse_args()
    run(args.run,allow_unverified_metadata=args.allow_unverified_metadata)
