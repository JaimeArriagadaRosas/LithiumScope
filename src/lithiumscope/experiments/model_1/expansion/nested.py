"""Nested spatial development of M1 transfer policies; no saved production model."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from uuid import uuid4

import numpy as np
import pandas as pd

from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.paths import RESULTS_DIR
from lithiumscope.core.scientific.regression_metrics import regression_metrics
from lithiumscope.datasets.management.local_sources import local_training_source
from lithiumscope.experiments.data_lab.candidate_audit import REFERENCE as HISTORICAL_REFERENCE, FOLD_SHA256
from lithiumscope.experiments.model_1.balanced.selection import balanced_positions
from lithiumscope.experiments.model_1.expansion.blend_policy import BASE, EXPERTS, POLICIES, combine, select_inner
from lithiumscope.experiments.model_1.expansion.sensitivity import REFERENCE, ALGORITHM, TRANSFER_WEIGHTS
from lithiumscope.experiments.model_1.expansion.sensitivity_data import load_evidence, historical_tokens, reserve, fold_pools
from lithiumscope.experiments.model_1.expansion.transfer_weights import fit_weighted
from lithiumscope.experiments.model_1.targeted.protocol import load_protocol
from lithiumscope.experiments.model_1.targeted.fit import _base_fit
from lithiumscope.experiments.model_1.targeted.policies import Arm
from lithiumscope.experiments.model_1.targeted.report import fold_metrics, summarize
from lithiumscope.experiments.shared.paired_comparison import paired_group_intervals
from lithiumscope.model_1.pipeline import prepare_training_data
from lithiumscope.resources.detection.device import detect_device
from lithiumscope.results.atomic_exports import write_legacy_evidence_json, write_table

SELECTED="K_inner_selected"


def run(run_path, *, allow_unverified_metadata=False):
    if not allow_unverified_metadata:
        raise ValueError("Explicit --allow-unverified-metadata required; research only")
    evidence,audit,features,pubs,links=load_evidence(run_path)
    data,splits,params,archived,identity=load_protocol(REFERENCE)
    x,y,groups,sources,_,numeric,categorical=data
    source=local_training_source("model_1")
    if identity["source_sha256"]!=evidence["source_sha256"] or file_sha256(source)!=evidence["source_sha256"]:
        raise ValueError("Historical source differs")
    if list(x.columns)!=evidence["features"] or numeric!=evidence["numeric"] or categorical!=evidence["categorical"]:
        raise ValueError("Frozen feature schema differs")
    fold_path=RESULTS_DIR/"model_1/runs"/HISTORICAL_REFERENCE/"tables/outer_fold_manifest.csv"
    if file_sha256(fold_path)!=FOLD_SHA256 or evidence.get("historical_fold_sha256")!=FOLD_SHA256:
        raise ValueError("Frozen fold identity differs")
    historical_folds=pd.read_csv(fold_path).sort_values("row_position")
    for fold,(_,te,_) in enumerate(splits,1):
        if not np.array_equal(te,np.flatnonzero(historical_folds.outer_fold.eq(fold))):
            raise ValueError("Reference RF TEST differs")
    prepared=prepare_training_data(source)
    if not np.array_equal(prepared.frame.index.astype(str),x.index.astype(str)):
        raise ValueError("Historical publication rows not aligned")
    historical_refs=historical_tokens(prepared.frame)
    cx=features[list(x.columns)].copy()
    for key in numeric:
        cx[key]=pd.to_numeric(cx[key],errors="raise").astype(float)
    for key in categorical:
        cx[key]=cx[key].astype(object).where(cx[key].notna(),np.nan)
    cy=pd.to_numeric(features.Li_icpms,errors="raise")
    cg=features.spatial_group.astype(str)
    cs=features.source_dataset.astype(str)
    if not np.isfinite(cy).all() or cy.lt(0).any():
        raise ValueError("Invalid candidate truth")
    held,pool=reserve(audit,pubs,groups,historical_refs)
    high=audit.screening_state.eq("high_transfer_review").to_numpy()
    h=np.flatnonzero(held)
    seed=identity["seed"]
    device=detect_device(prefer_gpu=False)
    root=RESULTS_DIR/"generalization/model_1_expansion_nested"
    output=root/(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")+"_"+uuid4().hex[:8])
    output.mkdir(parents=True,exist_ok=False)
    manifest={**identity,**links,"state":"running","scope":"nested_transfer_development_reuses_outer_folds",
        "metadata_certified":False,"publication_independence_certified":False,
        "independent_holdout":False,"release_eligible":False,"promotion":False,"models_saved":False,
        "acquisition":False,"historical_fold_sha256":FOLD_SHA256,
        "policy_selection":"inner_oof_guarded_rmse_mae_mean_fold_rmse_low_mid_band",
        "policies":POLICIES,"reserved_rows":len(h),"pool_rows":int(pool.sum()),
        "planned_inner_fits":3*sum(len(inner) for _,_,inner in splits),
        "max_outer_fits":2*len(splits),"historical_publication_token_rows":sum(bool(v) for v in historical_refs)}
    write_legacy_evidence_json(output/"manifest.json",manifest)
    predictions,metrics,ref_rows,reserved,trace,plans,choices,inner_evidence,inner_scores=[],[],[],[],[],[],[],[],[]
    attempted=0

    def fit_partition(fold,phase,tr,blocked,experts):
        nonlocal attempted
        if np.intersect1d(tr,blocked).size or set(groups.iloc[tr])&set(groups.iloc[blocked]):
            raise ValueError("Historical TRAIN overlaps excluded partition")
        blocked_refs=set().union(*(historical_refs[i] for i in blocked))
        positions=(fold_pools(audit,cx,x.iloc[tr],numeric,pool,pubs,set(groups.iloc[blocked]),blocked_refs)["D4"]
                   if experts else np.array([],dtype=int))
        if experts and (not len(positions) or not high[positions].any()):
            raise ValueError("No high-domain TRAIN extension in "+phase)
        keep=tr[balanced_positions(x.iloc[tr],y.iloc[tr],groups.iloc[tr],sources.iloc[tr],2700,seed)]
        bx,by,bg,bs=[v.iloc[keep].copy() for v in (x,y,groups,sources)]
        attempted+=1
        base,info=_base_fit(ALGORITHM,Arm("control"),device,seed,params[ALGORITHM,fold],
                            bx,by,bg,bs,numeric,categorical)
        trace.append({"fold":fold,"phase":phase,"model":BASE,**info})
        fits={BASE:base}
        if experts:
            ids=pd.Index("candidate:"+audit.candidate_id.iloc[positions].astype(str))
            additions=[v.iloc[positions].set_axis(ids) for v in (cx,cy,cg,cs)]
            tx,ty,tg,ts=[pd.concat([a,b]) for a,b in zip((bx,by,bg,bs),additions,strict=True)]
            if not tx.index.is_unique or set(tg)&set(groups.iloc[blocked]):
                raise ValueError("Expanded TRAIN overlaps excluded partition")
            if np.intersect1d(positions,h).size or any(pubs[i]&blocked_refs for i in positions):
                raise ValueError("Expanded TRAIN overlaps reserved/publication exclusion")
            mask=pd.Series(np.r_[np.zeros(len(keep),dtype=bool),high[positions]],index=tx.index)
            for expert in experts:
                attempted+=1
                predict,info=fit_weighted(device,seed,params[ALGORITHM,fold],tx,ty,tg,numeric,categorical,
                                           mask,TRANSFER_WEIGHTS[expert])
                fits[expert]=predict
                trace.append({"fold":fold,"phase":phase,"model":expert,**info})
        scale=float(y.iloc[tr].quantile(.75))
        plans.append({"fold":fold,"phase":phase,"historical_train":keep.tolist(),
            "excluded_historical":blocked.tolist(),"candidate_train_ids":audit.candidate_id.iloc[positions].tolist(),
            "models":[BASE,*experts],"train_scale":scale})
        return fits,scale

    def scores(fits,frame):
        result={name:np.maximum(np.asarray(predict(frame),float),0.) for name,predict in fits.items()}
        if any(v.shape!=(len(frame),) or not np.isfinite(v).all() for v in result.values()):
            raise ValueError("Invalid model predictions")
        return result

    def apply(values,policy,scale):
        expert=POLICIES[policy][0]
        return combine(values[BASE],values.get(expert),policy,scale)

    def rows(policy,fold,tr,te,values):
        maximum=float(y.iloc[tr].max())
        return [{"algorithm":ALGORITHM,"policy":policy,"fold":fold,"position":int(i),
            "row_id":str(x.index[i]),"group":str(groups.iloc[i]),"source":str(sources.iloc[i]),
            "truth":float(y.iloc[i]),"prediction":float(v),"train_max":maximum,
            "actual_above_train_max":bool(y.iloc[i]>maximum)} for i,v in zip(te,values,strict=True)]

    def checkpoint():
        write_table(output/"predictions.csv",pd.DataFrame(predictions))
        write_table(output/"fold_metrics.csv",pd.DataFrame(metrics))
        write_table(output/"inner_predictions.csv",pd.DataFrame(inner_evidence))
        write_table(output/"inner_scores.csv",pd.DataFrame(inner_scores))
        write_table(output/"reserved_predictions.csv",pd.DataFrame(reserved,columns=["fold","policy","candidate_id","truth","prediction"]))
        write_legacy_evidence_json(output/"selection.json",choices)
        write_legacy_evidence_json(output/"partitions.json",plans)
        write_legacy_evidence_json(output/"fit_trace.json",trace)

    try:
        for fold,(tr,te,inner) in enumerate(splits,1):
            internal=[]
            for number,(a,b) in enumerate(inner,1):
                if set(a)|set(b)!=set(tr) or np.intersect1d(a,b).size:
                    raise ValueError("Invalid inner partition")
                excluded=np.sort(np.r_[te,b])
                print(f"Outer {fold}/5 | inner {number}/{len(inner)} | B/H/I",flush=True)
                fits,scale=fit_partition(fold,f"inner_{number}",a,excluded,EXPERTS)
                values=scores(fits,x.iloc[b])
                for policy in POLICIES:
                    internal.extend({"position":int(i),"inner_fold":number,"truth":float(y.iloc[i]),
                        "policy":policy,"prediction":float(v)} for i,v in zip(b,apply(values,policy,scale),strict=True))
                del fits
            table=pd.DataFrame(internal)
            if set(table.loc[table.policy.eq(BASE),"position"])!=set(tr):
                raise ValueError("Inner OOF coverage differs")
            policy,detail=select_inner(table)
            inner_evidence.extend({"fold":fold,**row} for row in internal)
            inner_scores.extend({"fold":fold,**row} for row in detail)
            choices.append({"fold":fold,"selected_policy":policy,"eligible_policies":sum(r["guard_pass"] for r in detail)})
            expert=POLICIES[policy][0]
            print(f"Outer {fold}/5 | selected={policy}",flush=True)
            fits,scale=fit_partition(fold,"outer",tr,te,(expert,) if expert else ())
            test_values=scores(fits,x.iloc[te])
            train_values=scores(fits,x.iloc[tr])
            held_values=scores(fits,cx.iloc[h]) if len(h) else {}
            for label,chosen in ((BASE,BASE),(SELECTED,policy)):
                test=apply(test_values,chosen,scale)
                train=apply(train_values,chosen,scale)
                metrics.append({"algorithm":ALGORITHM,"policy":label,"fold":fold,
                    "selected_policy":chosen,**fold_metrics(y.iloc[te].to_numpy(float),test,y.iloc[tr].to_numpy(float),train)})
                predictions.extend(rows(label,fold,tr,te,test))
                if len(h):
                    reserved.extend({"fold":fold,"policy":label,"candidate_id":audit.candidate_id.iloc[i],
                        "truth":float(cy.iloc[i]),"prediction":float(v)}
                        for i,v in zip(h,apply(held_values,chosen,scale),strict=True))
            ref_rows.extend(rows("archived",fold,tr,te,archived[ALGORITHM][te]))
            del fits
            checkpoint()
        ref=pd.DataFrame(ref_rows)
        pred=pd.DataFrame(predictions)
        write_table(output/"archived_predictions.csv",ref)
        if any(len(part)!=len(x) or part.position.duplicated().any() for _,part in pred.groupby("policy")):
            raise ValueError("Outer OOF coverage differs")
        manifest["summary"]=summarize(output,pd.DataFrame(metrics),pred,len(x),{ALGORITHM:ref})
        write_table(output/"paired_vs_balanced.csv",pd.DataFrame(paired_group_intervals(
            pred.loc[pred.policy.eq(BASE)],pred.loc[pred.policy.eq(SELECTED)],kind="regression",repeats=1000)))
        bands=pred.copy()
        bands["band"]=pd.cut(bands.truth,[-np.inf,5,20,50,100,696,1000,np.inf]).astype(str)
        write_table(output/"band_metrics.csv",pd.DataFrame([{"policy":label,"band":band,"rows":len(part),
            **regression_metrics(part.truth,part.prediction).to_dict()} for (label,band),part in bands.groupby(["policy","band"])]))
        if reserved:
            rr=pd.DataFrame(reserved)
            result=[]
            for label,part in rr.groupby("policy"):
                if part.fold.nunique()!=len(splits) or len(part)!=len(h)*len(splits):
                    raise ValueError("Reserved prediction coverage differs")
                mean=part.groupby("candidate_id").agg(truth=("truth","first"),prediction=("prediction","mean"))
                result.append({"policy":label,"rows":len(mean),**regression_metrics(mean.truth,mean.prediction).to_dict()})
            write_table(output/"reserved_metrics.csv",pd.DataFrame(result))
        manifest["state"]="complete"
    except BaseException as exc:
        manifest["state"]="interrupted"
        manifest["error"]=str(exc)
        raise
    finally:
        manifest["attempted_fits"]=attempted
        manifest["selected_policies"]=choices
        manifest["outputs"]={p.name:file_sha256(p) for p in output.iterdir() if p.is_file() and p.name!="manifest.json"}
        write_legacy_evidence_json(output/"manifest.json",manifest)
    print(pd.DataFrame(manifest["summary"]).to_string(index=False))
    print(pd.DataFrame(choices).to_string(index=False))
    print(f"Estado: {manifest['state']} | Evidencias: {output}",flush=True)
    return output


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",required=True)
    parser.add_argument("--allow-unverified-metadata",action="store_true")
    args=parser.parse_args()
    run(args.run,allow_unverified_metadata=args.allow_unverified_metadata)
