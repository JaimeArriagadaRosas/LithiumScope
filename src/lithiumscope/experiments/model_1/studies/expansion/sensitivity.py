"""Fixed expansion sensitivity experiment, no promotion or saved estimators."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.paths import RESULTS_DIR
from lithiumscope.core.scientific.regression_metrics import regression_metrics
from lithiumscope.datasets.management.sources.local import local_training_source
from lithiumscope.experiments.model_1.treatments.balancing import balanced_positions
from lithiumscope.experiments.data_lab.screening.audit import REFERENCE as HISTORICAL_REFERENCE, FOLD_SHA256
from lithiumscope.experiments.model_1.data_expansion.sensitivity_data import load_evidence, historical_tokens, reserve, fold_pools
from lithiumscope.experiments.model_1.treatments.transfer_weights import fit_weighted
from lithiumscope.experiments.model_1.protocols.evidence import load_protocol
from lithiumscope.experiments.model_1.training_methods.fitting import _base_fit
from lithiumscope.experiments.model_1.treatments.policies import Arm
from lithiumscope.experiments.model_1.evaluation.targeted_metrics import fold_metrics, summarize
from lithiumscope.experiments.shared.paired_comparison import paired_group_intervals
from lithiumscope.model_1.pipeline import prepare_training_data
from lithiumscope.resources.detection.device import detect_device
from lithiumscope.results.atomic_exports import write_legacy_evidence_json, write_table

REFERENCE="training_20261125_115932_m0300_826123c1"
ALGORITHM="random_forest"
ARMS={"A_control":None,"B_balanced":None,"C_convergent":"D1","D_high":"D2",
      "E_combined":"D3","F_broader":"D4"}


TRANSFER_ARMS={"G_no_high":"D4","H_weight_010":"D4","I_weight_025":"D4","J_weight_050":"D4"}
TRANSFER_WEIGHTS={"H_weight_010":.1,"I_weight_025":.25,"J_weight_050":.5}


FAMILY_ARMS={"B_balanced":None,"H_weight_010":"D4","I_weight_025":"D4"}
FAMILIES=("random_forest","catboost","hist_gradient_boosting")


def run(run_path, *, allow_unverified_metadata=False, transfer_refinement=False, algorithm=ALGORITHM):
    if not allow_unverified_metadata:
        raise ValueError("Explicit --allow-unverified-metadata required; research sensitivity only")
    if algorithm not in FAMILIES or (algorithm!=ALGORITHM and transfer_refinement):
        raise ValueError("Unsupported algorithm or incompatible campaign")
    family=algorithm!=ALGORITHM
    arms=FAMILY_ARMS if family else (TRANSFER_ARMS if transfer_refinement else ARMS)
    planned=len(arms)*5
    evidence,audit,features,pubs,links=load_evidence(run_path)
    data,splits,params,archived,identity=load_protocol(REFERENCE)
    x,y,groups,sources,_,numeric,categorical=data
    source=local_training_source("model_1")
    if identity["source_sha256"]!=evidence["source_sha256"] or file_sha256(source)!=evidence["source_sha256"]:
        raise ValueError("Historical cohort hash differs")
    if list(x.columns)!=evidence["features"] or numeric!=evidence["numeric"] or categorical!=evidence["categorical"]:
        raise ValueError("Frozen predictor schema differs")
    fold_path=RESULTS_DIR/"model_1/runs"/HISTORICAL_REFERENCE/"tables/outer_fold_manifest.csv"
    if file_sha256(fold_path)!=FOLD_SHA256 or evidence.get("historical_fold_sha256")!=FOLD_SHA256:
        raise ValueError("Frozen fold identity differs")
    historical_folds=pd.read_csv(fold_path).sort_values("row_position")
    for fold,(_,te,_) in enumerate(splits,1):
        if not np.array_equal(te,np.flatnonzero(historical_folds.outer_fold.eq(fold))):
            raise ValueError("Reference RF uses different historical TEST")
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
        raise ValueError("Candidate truth invalid")
    held,pool=reserve(audit,pubs,groups,historical_refs)
    plans=[]
    for fold,(tr,te,_) in enumerate(splits,1):
        blocked_refs=set().union(*(historical_refs[i] for i in te))
        chosen=fold_pools(audit,cx,x.iloc[tr],numeric,pool,pubs,set(groups.iloc[te]),blocked_refs)
        keep=tr[balanced_positions(x.iloc[tr],y.iloc[tr],groups.iloc[tr],sources.iloc[tr],2700,identity["seed"])]
        plans.append((fold,tr,te,keep,chosen))
    high_mask=audit.screening_state.eq("high_transfer_review").to_numpy()
    def arm_positions(policy,cohort,chosen):
        positions=chosen[cohort] if cohort else np.array([],dtype=int)
        if policy=="G_no_high":
            positions=positions[~high_mask[positions]]
        return positions
    campaign="model_1_expansion_transfer" if transfer_refinement else "model_1_expansion_sensitivity"
    if family:
        campaign="model_1_expansion_families/"+algorithm
    destination=RESULTS_DIR/"generalization"/campaign/(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")+"_"+uuid4().hex[:8])
    destination.mkdir(parents=True,exist_ok=False)
    selection=audit[["candidate_id","screening_state","source_dataset","Li_icpms","spatial_group","source_publication",
                     "bibliography_resolved","sensitivity_eligible","historical_exact_overlap","metadata_review_required"]].copy()
    selection["partition"]=np.select([held,pool,audit.sensitivity_eligible],["reserved","train_pool","quarantine"],default="excluded")
    write_table(destination/"candidate_partition.csv",selection)
    plan_rows=[]
    for fold,tr,te,keep,chosen in plans:
        for arm,cohort in arms.items():
            positions=arm_positions(arm,cohort,chosen)
            plan_rows.append({"fold":fold,"policy":arm,"historical_train":(tr if arm=="A_control" else keep).tolist(),
                "historical_test":te.tolist(),"candidate_train_ids":audit.candidate_id.iloc[positions].tolist(),
                "candidate_train_count":len(positions)})
    write_legacy_evidence_json(destination/"partitions.json",plan_rows)
    manifest={**identity,**links,"state":"running","scope":"metadata_uncertified_sensitivity_reused_outer_folds",
              "independent_holdout":False,"release_eligible":False,"metadata_certified":False,"selection":False,
              "promotion":False,"models_saved":False,"acquisition":False,"planned_fits":planned,"transfer_refinement":transfer_refinement,
              "reserved_rows":int(held.sum()),"pool_rows":int(pool.sum()),
              "historical_publication_token_rows":sum(bool(v) for v in historical_refs),
              "candidate_eligible":int(audit.sensitivity_eligible.sum()),"publication_independence_certified":False,"historical_fold_sha256":FOLD_SHA256,
              "arms":arms,"algorithm":algorithm,"transfer_weights":{k:v for k,v in TRANSFER_WEIGHTS.items() if k in arms}}
    write_legacy_evidence_json(destination/"manifest.json",manifest)
    device=detect_device(prefer_gpu=False)
    predictions,metrics,reserved,trace,errors,ref_rows=[],[],[],[],[],[]
    attempted=0
    def rows(policy,fold,positions,scores,train_max):
        return [{"algorithm":algorithm,"policy":policy,"fold":fold,"position":int(i),"row_id":str(x.index[i]),
                 "group":str(groups.iloc[i]),"source":str(sources.iloc[i]),"truth":float(y.iloc[i]),
                 "prediction":float(score),"train_max":float(train_max),
                 "actual_above_train_max":bool(y.iloc[i]>train_max)}
                for i,score in zip(positions,scores,strict=True)]
    try:
        for fold,tr,te,keep,chosen in plans:
            ref_rows.extend(rows("archived",fold,te,archived[algorithm][te],y.iloc[tr].max()))
            for policy,cohort in arms.items():
                pos=arm_positions(policy,cohort,chosen)
                hist=tr if policy=="A_control" else keep
                if cohort and not len(pos):
                    errors.append({"fold":fold,"policy":policy,"reason":"No eligible new rows; no fit"})
                    continue
                print(f"Fold {fold}/5 | {policy} | historical={len(hist)} new={len(pos)}",flush=True)
                try:
                    bx,by,bg,bs=[v.iloc[hist].copy() for v in (x,y,groups,sources)]
                    if len(pos):
                        ids=pd.Index("candidate:"+audit.candidate_id.iloc[pos].astype(str))
                        additions=[v.iloc[pos].set_axis(ids) for v in (cx,cy,cg,cs)]
                        bx,by,bg,bs=[pd.concat([a,b]) for a,b in zip((bx,by,bg,bs),additions,strict=True)]
                    if not bx.index.is_unique or set(bg)&set(groups.iloc[te]):
                        raise ValueError("TRAIN identity or spatial overlap")
                    attempted+=1
                    if policy in TRANSFER_WEIGHTS:
                        high=pd.Series(np.r_[np.zeros(len(hist),dtype=bool),high_mask[pos]],index=bx.index)
                        if not high.any():
                            raise ValueError("No high-domain samples to attenuate")
                        predict,info=fit_weighted(device,identity["seed"],params[algorithm,fold],
                            bx,by,bg,numeric,categorical,high,TRANSFER_WEIGHTS[policy],algorithm=algorithm)
                    else:
                        predict,info=_base_fit(algorithm,Arm("control"),device,identity["seed"],
                                              params[algorithm,fold],bx,by,bg,bs,numeric,categorical)
                    score=np.maximum(np.asarray(predict(x.iloc[te]),float),0.)
                    train_score=np.maximum(np.asarray(predict(x.iloc[tr]),float),0.)
                    if not np.isfinite(score).all() or not np.isfinite(train_score).all():
                        raise ValueError("Nonfinite predictions")
                    values=fold_metrics(y.iloc[te].to_numpy(float),score,y.iloc[tr].to_numpy(float),train_score)
                    metrics.append({"algorithm":algorithm,"policy":policy,"fold":fold,"new_rows":len(pos),
                        "actual_fit_target_max":float(by.max()),**values})
                    # Keep support stratum fixed to historical TRAIN across arms for fair comparison.
                    predictions.extend(rows(policy,fold,te,score,y.iloc[tr].max()))
                    trace.append({"fold":fold,"policy":policy,"historical_fit_rows":len(hist),
                                  "new_fit_rows":len(pos),"actual_train_max":float(by.max()),**info})
                    if held.any():
                        h=np.flatnonzero(held)
                        hs=np.maximum(np.asarray(predict(cx.iloc[h]),float),0.)
                        if not np.isfinite(hs).all():
                            raise ValueError("Nonfinite reserved predictions")
                        reserved.extend({"fold":fold,"policy":policy,"candidate_id":audit.candidate_id.iloc[i],
                                         "truth":float(cy.iloc[i]),"prediction":float(s)} for i,s in zip(h,hs,strict=True))
                except Exception as exc:
                    errors.append({"fold":fold,"policy":policy,"reason":str(exc)})
                    print(f"[ERROR] {exc}",flush=True)
            write_table(destination/"predictions.csv",pd.DataFrame(predictions))
            write_table(destination/"fold_metrics.csv",pd.DataFrame(metrics))
            write_table(destination/"reserved_predictions.csv",pd.DataFrame(reserved,columns=["fold","policy","candidate_id","truth","prediction"]))
            write_legacy_evidence_json(destination/"fit_trace.json",trace)
            write_legacy_evidence_json(destination/"errors.json",errors)
        ref=pd.DataFrame(ref_rows)
        write_table(destination/"archived_predictions.csv",ref)
        pred=pd.DataFrame(predictions)
        manifest["summary"]=summarize(destination,pd.DataFrame(metrics),pred,len(x),{algorithm:ref}) if len(pred) else []
        if family and not errors and attempted==planned:
            baseline=pred.loc[pred.policy.eq("B_balanced")]
            comparisons=[]
            for policy in ("H_weight_010","I_weight_025"):
                for item in paired_group_intervals(baseline,pred.loc[pred.policy.eq(policy)],kind="regression",repeats=1000):
                    comparisons.append({"policy":policy,**item})
            write_table(destination/"paired_vs_balanced.csv",pd.DataFrame(comparisons))
        if len(pred):
            bands=pred.copy()
            bands["band"]=pd.cut(bands.truth,[-np.inf,5,20,50,100,696,1000,np.inf]).astype(str)
            output=[]
            for (policy,band),part in bands.groupby(["policy","band"]):
                output.append({"policy":policy,"band":band,"rows":len(part),
                               **regression_metrics(part.truth,part.prediction).to_dict()})
            write_table(destination/"band_metrics.csv",pd.DataFrame(output))
        if reserved:
            rr=pd.DataFrame(reserved)
            output=[]
            for policy,part in rr.groupby("policy"):
                if part.fold.nunique()!=5 or len(part)!=5*int(held.sum()):
                    continue
                mean=part.groupby("candidate_id").agg(truth=("truth","first"),prediction=("prediction","mean"))
                output.append({"policy":policy,"rows":len(mean),**regression_metrics(mean.truth,mean.prediction).to_dict()})
            write_table(destination/"reserved_metrics.csv",pd.DataFrame(output))
        manifest["state"]="complete" if not errors and attempted==planned else "partial"
    except BaseException:
        manifest["state"]="interrupted"
        raise
    finally:
        manifest["attempted_fits"]=attempted
        manifest["errors"]=errors
        manifest["outputs"]={p.name:file_sha256(p) for p in destination.iterdir() if p.is_file() and p.name!="manifest.json"}
        write_legacy_evidence_json(destination/"manifest.json",manifest)
    print(pd.DataFrame(manifest["summary"]).to_string(index=False))
    print(f"Estado: {manifest['state']} | Evidencias: {destination}",flush=True)
    return destination


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",required=True)
    parser.add_argument("--allow-unverified-metadata",action="store_true")
    parser.add_argument("--transfer-refinement",action="store_true")
    parser.add_argument("--algorithm",choices=FAMILIES,default=ALGORITHM)
    args=parser.parse_args()
    run(args.run,allow_unverified_metadata=args.allow_unverified_metadata,transfer_refinement=args.transfer_refinement,algorithm=args.algorithm)
