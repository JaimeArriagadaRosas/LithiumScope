"""M2-only follow-up. No calibration on TEST, saved models, selection or promotion."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from uuid import uuid4

import numpy as np
import pandas as pd

from lithiumscope.core.paths import RESULTS_DIR
from lithiumscope.core.reproducibility import runtime_fingerprint
from lithiumscope.experiments.generalization.partitions import training_target
from lithiumscope.experiments.model_2.refinement_evidence import load_protocol
from lithiumscope.experiments.model_2.refinement_policies import POLICIES, fit_policy
from lithiumscope.experiments.model_2.refinement_metrics import ranking, summarize
from lithiumscope.resources.detection.device import detect_device
from lithiumscope.results.atomic_exports import write_table, write_legacy_evidence_json


def evaluate_refinement(run_id, *, algorithms=None, output_root=None, device=None):
    data,splits,cfg,evidence=load_protocol(run_id)
    x,y,groups,_,eligible,_,_=data
    selected=tuple(POLICIES) if algorithms is None else tuple(algorithms)
    if not selected or len(set(selected))!=len(selected) or not set(selected)<=set(POLICIES):
        raise ValueError("Unknown or repeated M2 algorithm")
    device=device or detect_device(prefer_gpu=False)
    root=Path(output_root) if output_root is not None else RESULTS_DIR/"generalization"/"model_2_refinement"
    destination=root/(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")+"_"+uuid4().hex[:8])
    destination.mkdir(parents=True,exist_ok=False)
    total=sum(len(POLICIES[a]) for a in selected)*len(splits)
    manifest={**evidence,"state":"running","scope":"development_reuses_screening_outer_folds",
        "independent_holdout":False,"automatic_selection":False,"promotion":False,
        "fitted_models_saved":False,"runtime":runtime_fingerprint(),"planned_fits":total,
        "policies":{a:[asdict(p) for p in POLICIES[a]] for a in selected},
        "ranking":"mean spatial fold metrics; raw SVM margins are not probabilities"}
    write_legacy_evidence_json(destination/"manifest.json",manifest)
    metrics,predictions,provenance=[],[],[]
    started=monotonic()
    print(f"\nM2 — REFINAMIENTO DIRIGIDO | {total} ajustes previstos",flush=True)
    try:
        for algorithm in selected:
            for fold,(tr,te,_) in enumerate(splits,1):
                fit,labels,cut=training_target("model_2",data,cfg,tr)
                truth=(y.iloc[te]>=cut).astype(int)
                if set(groups.iloc[fit])&set(groups.iloc[te]):
                    raise ValueError("Spatial overlap")
                for policy in POLICIES[algorithm]:
                    began=monotonic()
                    row={"algorithm":algorithm,"policy":policy.name,"fold":fold,"state":"error",
                        "threshold_ppm":cut,"train_rows":len(fit),"test_rows":len(te),
                        "test_positive":int(truth.sum()),"test_groups":int(groups.iloc[te].nunique())}
                    print(f"[{len(metrics)+1}/{total}] {algorithm} | fold {fold}/{len(splits)} | {policy.name}",flush=True)
                    try:
                        score,train_score,detail,trace=fit_policy(algorithm,policy,device,42+fold,
                            x.iloc[fit],labels,groups.iloc[fit],x.iloc[te])
                        allowed=set(x.index[fit])
                        if any(str(t[k]) not in allowed for t in trace for k in ("row","parent_a","parent_b") if k in t):
                            raise ValueError("Signature provenance outside eligible TRAIN")
                        result=ranking(truth,score,x.iloc[te],groups.iloc[te])
                        train=ranking(labels,train_score,x.iloc[fit],groups.iloc[fit])
                        row.update(result)
                        row.update({"train_"+k:v for k,v in train.items()})
                        row.update(detail,state="ok")
                        predictions.extend({"algorithm":algorithm,"policy":policy.name,"fold":fold,
                            "position":int(j),"row_id":str(x.index[j]),"group":str(groups.iloc[j]),
                            "truth":int(t),"score":float(s),"score_kind":detail["score_kind"]}
                            for j,t,s in zip(te,truth,score,strict=True))
                        provenance.extend({"algorithm":algorithm,"policy":policy.name,"fold":fold,**t} for t in trace)
                        print(f"  [OK] AUC={result['roc_auc']:.4f} | AP={result['ap']:.4f} | TRAIN={train['roc_auc']:.4f}",flush=True)
                    except Exception as exc:
                        row["reason"]=str(exc)
                        print(f"  [ERROR] {exc}",flush=True)
                    row["seconds"]=monotonic()-began
                    metrics.append(row)
                    write_table(destination/"fold_metrics.csv",pd.DataFrame(metrics))
                    write_table(destination/"predictions.csv",pd.DataFrame(predictions))
                    write_table(destination/"signature_provenance.csv",pd.DataFrame(provenance,
                        columns=["algorithm","policy","fold","row","parent_a","action","weight","signature","group"]))
        table=pd.DataFrame(metrics)
        pred=pd.DataFrame(predictions)
        summaries,deltas=[],[]
        for algorithm in selected:
            ref=table.loc[(table.algorithm==algorithm)&(table.policy=="reference")].sort_values("fold")
            for policy in POLICIES[algorithm]:
                folds=table.loc[(table.algorithm==algorithm)&(table.policy==policy.name)].sort_values("fold")
                if len(folds)!=len(splits) or not folds.state.eq("ok").all():
                    summaries.append({"algorithm":algorithm,"policy":policy.name,"state":"partial",
                                      "completed_folds":int(folds.state.eq("ok").sum())})
                    continue
                z=pred.loc[(pred.algorithm==algorithm)&(pred.policy==policy.name)]
                if len(z)!=len(x) or z.position.duplicated().any():
                    raise ValueError("Incomplete OOF coverage")
                row={"algorithm":algorithm,"policy":policy.name,"state":"complete",**summarize(folds)}
                if policy.name!="reference" and len(ref)==len(splits) and ref.state.eq("ok").all():
                    for key in ("roc_auc","ap","signature_auc","signature_ap","group_weighted_auc","group_weighted_ap"):
                        gain=folds[key].to_numpy()-ref[key].to_numpy()
                        row["delta_"+key]=float(gain.mean())
                        if key=="roc_auc":
                            row["auc_folds_improved"]=int((gain>0).sum())
                        for f,value in zip(folds.fold,gain,strict=True):
                            deltas.append({"algorithm":algorithm,"policy":policy.name,"fold":int(f),
                                           "metric":key,"delta":float(value)})
                summaries.append(row)
        write_table(destination/"summary.csv",pd.DataFrame(summaries))
        write_table(destination/"paired_fold_deltas.csv",pd.DataFrame(deltas))
        manifest["summary"]=summaries
        manifest["state"]="complete" if len(metrics)==total and all(r["state"]=="ok" for r in metrics) else "partial"
    except BaseException:
        manifest["state"]="interrupted"
        raise
    finally:
        manifest["completed_fits"]=len(metrics)
        manifest["successful_fits"]=sum(r["state"]=="ok" for r in metrics)
        manifest["elapsed_seconds"]=monotonic()-started
        write_legacy_evidence_json(destination/"manifest.json",manifest)
    print(f"\nEstado: {manifest['state']} | Evidencias: {destination/'manifest.json'}",flush=True)
    return destination


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-run",required=True)
    parser.add_argument("--algorithm",choices=list(POLICIES),action="append")
    args=parser.parse_args()
    evaluate_refinement(args.previous_run,algorithms=args.algorithm)


if __name__=="__main__":
    main()
