"""Targeted M1 follow-up: paired developmental evaluation, no model publication."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from time import monotonic
from uuid import uuid4

import numpy as np
import pandas as pd

from lithiumscope.core.paths import RESULTS_DIR
from lithiumscope.core.scientific.regression_metrics import regression_metrics
from lithiumscope.core.reproducibility import runtime_fingerprint
from lithiumscope.experiments.model_1.studies.refinement.evidence import load_protocol
from lithiumscope.experiments.model_1.studies.refinement.policies import POLICIES, fit_policy
from lithiumscope.experiments.shared.paired_comparison import paired_group_intervals
from lithiumscope.resources.detection.device import detect_device
from lithiumscope.results.atomic_exports import write_json, write_table, write_legacy_evidence_json


def evaluate_refinement(run_id, *, algorithms=None, output_root=None, device=None):
    data, splits, evidence = load_protocol(run_id)
    x,y,groups,sources,eligible,numeric,categorical = data
    selected = tuple(POLICIES) if algorithms is None else tuple(algorithms)
    if not selected or len(set(selected))!=len(selected) or not set(selected)<=set(POLICIES):
        raise ValueError("Unknown or duplicated M1 families")
    device=device or detect_device(prefer_gpu=False)
    root=output_root if output_root is not None else RESULTS_DIR/"generalization"/"model_1_refinement"
    destination=root/(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")+"_"+uuid4().hex[:8])
    destination.mkdir(parents=True,exist_ok=False)
    manifest={**evidence,"state":"running","scope":"development_reuses_screening_outer_folds",
              "independent_holdout":False,"automatic_selection":False,"promotion":False,
              "fitted_models_saved":False,"runtime":runtime_fingerprint(),
              "policies":{a:[asdict(p) for p in POLICIES[a]] for a in selected}}
    total=sum(len(POLICIES[a]) for a in selected)*len(splits)
    manifest["planned_fits"]=total
    write_legacy_evidence_json(destination/"manifest.json",manifest)
    metrics, predictions, augmentation=[],[],[]
    started=monotonic()
    print(f"\nM1 — REFINAMIENTO DIRIGIDO | {total} ajustes previstos",flush=True)
    try:
        for algorithm in selected:
            for fold,(tr,te,_) in enumerate(splits,1):
                fit=tr[eligible[tr]]
                if set(groups.iloc[fit]) & set(groups.iloc[te]):
                    raise ValueError("Spatial overlap")
                q75=float(y.iloc[fit].quantile(.75))
                maximum=float(y.iloc[fit].max())
                truth=y.iloc[te].to_numpy(float)
                high=truth>=q75
                for policy in POLICIES[algorithm]:
                    begin=monotonic()
                    row={"algorithm":algorithm,"policy":policy.name,"fold":fold,"state":"error",
                         "train_rows":len(fit),"test_rows":len(te),"train_q75":q75,"train_max":maximum,
                         "test_above_train_max":int((truth>maximum).sum())}
                    print(f"[{len(metrics)+1}/{total}] {algorithm} | fold {fold}/{len(splits)} | {policy.name}",flush=True)
                    try:
                        scores,train_scores,negative,above,trace,fit_count=fit_policy(
                            algorithm,policy,device,42+fold,x.iloc[fit],y.iloc[fit],groups.iloc[fit],
                            sources.iloc[fit],numeric,categorical,x.iloc[te])
                        train_ids=set(x.index[fit])
                        if any(str(entry[key]) not in train_ids for entry in trace for key in ("row","parent_a","parent_b") if key in entry):
                            raise ValueError("Resampling provenance outside TRAIN")
                        if policy.nonnegative and (scores<0).any():
                            raise ValueError("Physical projection failed")
                        result=regression_metrics(truth,scores).to_dict()
                        train=regression_metrics(y.iloc[fit],train_scores).to_dict()
                        row.update(result)
                        row.update({"train_"+k:v for k,v in train.items()})
                        row.update(state="ok",fit_rows=fit_count,raw_negative=int(negative.sum()),
                                   negative_predictions=int((scores<0).sum()),above_train_support=int(above.sum()),
                                   upper_bound_applied=int(above.sum()) if policy.bounded else 0,
                                   high_mae=float(np.abs(scores[high]-truth[high]).mean()) if high.any() else None,
                                   high_bias=float((scores[high]-truth[high]).mean()) if high.any() else None)
                        predictions.extend({"algorithm":algorithm,"policy":policy.name,"fold":fold,
                            "position":int(j),"row_id":str(x.index[j]),"group":str(groups.iloc[j]),
                            "source":str(sources.iloc[j]),"truth":float(t),"prediction":float(s),
                            "high_train_q75":bool(h),"actual_above_train_max":bool(t>maximum),
                            "raw_negative":bool(n),"above_train_support":bool(a)}
                            for j,t,s,h,n,a in zip(te,truth,scores,high,negative,above,strict=True))
                        augmentation.extend({"algorithm":algorithm,"policy":policy.name,"fold":fold,**entry} for entry in trace)
                        print(f"  [OK] RMSE={result['rmse']:.4f} | MAE={result['mae']:.4f} | TRAIN={train['rmse']:.4f}",flush=True)
                    except Exception as exc:
                        row["reason"]=str(exc)
                        print(f"  [ERROR] {exc}",flush=True)
                    row["seconds"]=monotonic()-begin
                    metrics.append(row)
                    write_table(destination/"fold_metrics.csv",pd.DataFrame(metrics))
                    write_table(destination/"predictions.csv",pd.DataFrame(predictions))
                    write_table(destination/"augmentation.csv",pd.DataFrame(augmentation,
                        columns=["algorithm","policy","fold","row","action","parent_a","parent_b","lambda","group","source"]))
        table=pd.DataFrame(metrics)
        pred=pd.DataFrame(predictions)
        summaries,intervals=[],[]
        for algorithm in selected:
            for policy in POLICIES[algorithm]:
                folds=table.loc[(table.algorithm==algorithm)&(table.policy==policy.name)]
                if len(folds)!=len(splits) or not folds.state.eq("ok").all():
                    summaries.append({"algorithm":algorithm,"policy":policy.name,"state":"partial",
                                      "completed_folds":int(folds.state.eq("ok").sum())})
                    continue
                z=pred.loc[(pred.algorithm==algorithm)&(pred.policy==policy.name)]
                if len(z)!=len(y) or z.position.duplicated().any():
                    raise ValueError("Refinement OOF coverage incomplete")
                global_metrics=regression_metrics(z.truth,z.prediction).to_dict()
                errors=(z.prediction-z.truth)**2
                summaries.append({"algorithm":algorithm,"policy":policy.name,"state":"complete",
                    "mean_fold_rmse":float(folds.rmse.mean()),"mean_fold_r2":float(folds.r2.mean()),
                    "worst_fold_rmse":float(folds.rmse.max()),"mean_train_rmse":float(folds.train_rmse.mean()),
                    "mean_train_r2":float(folds.train_r2.mean()),"negative_predictions":int(folds.negative_predictions.sum()),
                    "raw_negative":int(folds.raw_negative.sum()),"upper_bound_applied":int(folds.upper_bound_applied.sum()),
                    "high_mae":float(folds.high_mae.mean()),"high_bias":float(folds.high_bias.mean()),
                    "above_train_max_rmse":float(np.sqrt(errors[z.actual_above_train_max].mean())) if z.actual_above_train_max.any() else None,
                    **{"global_"+k:v for k,v in global_metrics.items()}})
                reference=pred.loc[(pred.algorithm==algorithm)&(pred.policy=="reference")]
                if policy.name!="reference" and len(reference)==len(y):
                    for result in paired_group_intervals(reference,z,kind="regression",repeats=500):
                        intervals.append({"algorithm":algorithm,"policy":policy.name,**result})
        write_table(destination/"summary.csv",pd.DataFrame(summaries))
        write_table(destination/"paired_intervals.csv",pd.DataFrame(intervals))
        manifest["state"]="complete" if len(metrics)==total and all(r["state"]=="ok" for r in metrics) else "partial"
        manifest["summary"]=summaries
        manifest["paired_intervals"]=intervals
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
