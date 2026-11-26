"""45 M1 configurations, nested spatial selection, no model publication."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from uuid import uuid4

import numpy as np
import pandas as pd

from lithiumscope.core.paths import RESULTS_DIR
from lithiumscope.core.reproducibility import runtime_fingerprint
from lithiumscope.experiments.model_1.targeted.fit import fit_arm
from lithiumscope.experiments.model_1.targeted.policies import ALGORITHMS, ARMS
from lithiumscope.experiments.model_1.targeted.protocol import load_protocol
from lithiumscope.experiments.model_1.targeted.report import fold_metrics, summarize
from lithiumscope.resources.detection.device import detect_device
from lithiumscope.results.atomic_exports import write_legacy_evidence_json, write_table


def select_arm(rows):
    """Use inner validation only. Guard MAE; rank by pooled RMSE then high bias."""
    control = next((r for r in rows if r["policy"] == "control"), None)
    if control is None:
        raise ValueError("Control unavailable; cannot select a treatment")
    allowed = [r for r in rows if r["mae"] <= control["mae"]]
    return min(allowed, key=lambda r:(r["rmse"], abs(r["high_bias"]), r["policy"]))["policy"]


def run_targeted(run_id, *, output_root=None, device=None, arms=ARMS, fit_function=fit_arm, campaign="model_1_targeted"):
    data, splits, params, archived, identity = load_protocol(run_id)
    x, y, groups, sources, _, numeric, categorical = data
    seed = identity["seed"]
    device = device or detect_device(prefer_gpu=False)
    root = output_root or RESULTS_DIR/"generalization"/campaign
    destination = root/(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")+"_"+uuid4().hex[:8])
    destination.mkdir(parents=True, exist_ok=False)
    planned = len(ALGORITHMS)*len(arms)*sum(len(inner)+1 for _,_,inner in splits)
    planned_calibration = len(ALGORITHMS)*sum(a.correction for a in arms)*sum(len(inner)+1 for _,_,inner in splits)*3
    manifest = {**identity,"state":"running","scope":"development_reuses_outer_folds",
                "independent_holdout":False,"promotion":False,"fitted_models_saved":False,
                "campaign":campaign,"runtime":runtime_fingerprint(),"arms":[asdict(a) for a in arms],
                "planned_evaluations":planned,"planned_calibration_fits":planned_calibration,"configurations":len(ALGORITHMS)*len(arms),
                "hyperparameters":"sealed_previous_outer_train_optimization_frozen_for_all_arms"}
    write_legacy_evidence_json(destination/"manifest.json", manifest)
    write_legacy_evidence_json(destination/"partitions.json", [{"fold":n,"train":tr.tolist(),"test":te.tolist(),
        "inner":[{"train":a.tolist(),"validation":b.tolist()} for a,b in inner]}
        for n,(tr,te,inner) in enumerate(splits,1)])
    metrics, predictions, inner_rows, selected, trace, errors = [], [], [], [], [], []
    print(f"M1: {planned} evaluaciones + {planned_calibration} ajustes de calibración previstos",flush=True)
    if campaign=="model_1_balanced":
        print("Los expertos añaden ajustes de enrutamiento y regresiones por régimen; se registran al completar.",flush=True)
    references = {}
    count = 0

    def prediction_rows(algorithm, policy, fold, test, scores):
        tr = splits[fold-1][0]
        maximum = float(y.iloc[tr].max())
        q75 = float(y.iloc[tr].quantile(.75))
        return [{"algorithm":algorithm,"policy":policy,"fold":fold,"position":int(j),
                 "row_id":str(x.index[j]),"group":str(groups.iloc[j]),"source":str(sources.iloc[j]),
                 "truth":float(y.iloc[j]),"prediction":float(score),
                 "train_max":maximum,"train_q75":q75,"actual_above_train_max":bool(y.iloc[j]>maximum)}
                for j,score in zip(test,scores,strict=True)]

    def evaluate(algorithm, arm, fold, phase, tr, te):
        nonlocal count
        count += 1
        if set(groups.iloc[tr]) & set(groups.iloc[te]):
            raise ValueError("Spatial TRAIN/validation overlap")
        print(f"[{count}/{planned}] {algorithm} | fold {fold} | {phase} | {arm.name}",flush=True)
        score, train_score, info = fit_function(algorithm,arm,device,seed,params[algorithm,fold],
            x.iloc[tr],y.iloc[tr],groups.iloc[tr],sources.iloc[tr],numeric,categorical,x.iloc[te])
        values = fold_metrics(y.iloc[te].to_numpy(float),score,y.iloc[tr].to_numpy(float),train_score)
        trace.append({"algorithm":algorithm,"policy":arm.name,"fold":fold,"phase":phase,**info})
        return score, values

    try:
        for algorithm in ALGORITHMS:
            reference_rows = []
            for fold,(tr,te,inner) in enumerate(splits,1):
                reference_rows.extend(prediction_rows(algorithm,"archived",fold,te,archived[algorithm][te]))
                scores_inner = []
                for arm in arms:
                    pieces, truths, biases = [], [], []
                    try:
                        for number,(a,b) in enumerate(inner,1):
                            score, values = evaluate(algorithm,arm,fold,f"inner_{number}",a,b)
                            inner_rows.append({"algorithm":algorithm,"policy":arm.name,"fold":fold,
                                               "inner_fold":number,**values})
                            pieces.extend(score)
                            truths.extend(y.iloc[b].to_numpy(float))
                            if np.isfinite(values["high_bias"]):
                                biases.append(values["high_bias"])
                        err = np.asarray(pieces)-np.asarray(truths)
                        scores_inner.append({"policy":arm.name,"rmse":float(np.sqrt(np.mean(err**2))),
                                             "mae":float(np.abs(err).mean()),
                                             "high_bias":float(np.mean(biases)) if biases else 0.})
                    except Exception as exc:
                        errors.append({"algorithm":algorithm,"fold":fold,"policy":arm.name,"phase":"inner","reason":str(exc)})
                        print(f"  [BLOQUEADO/ERROR] {exc}",flush=True)
                chosen = select_arm(scores_inner)
                selected.append({"algorithm":algorithm,"fold":fold,"policy":chosen,"inner_scores":scores_inner})
                for arm in arms:
                    try:
                        score, values = evaluate(algorithm,arm,fold,"outer",tr,te)
                        row = {"algorithm":algorithm,"policy":arm.name,"fold":fold,**values}
                        metrics.append(row)
                        predictions.extend(prediction_rows(algorithm,arm.name,fold,te,score))
                        if arm.name == chosen:
                            metrics.append({**row,"policy":"selected_inner"})
                            predictions.extend(prediction_rows(algorithm,"selected_inner",fold,te,score))
                    except Exception as exc:
                        errors.append({"algorithm":algorithm,"fold":fold,"policy":arm.name,"phase":"outer","reason":str(exc)})
                        print(f"  [BLOQUEADO/ERROR] {exc}",flush=True)
                write_table(destination/"fold_metrics.csv",pd.DataFrame(metrics))
                write_table(destination/"inner_metrics.csv",pd.DataFrame(inner_rows))
                write_table(destination/"predictions.csv",pd.DataFrame(predictions))
                write_legacy_evidence_json(destination/"selection.json",selected)
                write_legacy_evidence_json(destination/"treatment_evidence.json",trace)
                write_legacy_evidence_json(destination/"errors.json",errors)
            references[algorithm] = pd.DataFrame(reference_rows)
        write_table(destination/"archived_predictions.csv",pd.concat(references.values(),ignore_index=True))
        manifest["summary"] = summarize(destination,pd.DataFrame(metrics),pd.DataFrame(predictions),len(x),references)
        manifest["state"] = "complete" if not errors and count == planned else "partial"
    except BaseException:
        manifest["state"] = "interrupted"
        raise
    finally:
        manifest.update(attempted_evaluations=count,errors=errors,
                        calibration_fits=sum(item["calibration_fits"] for item in trace),
                        extra_regime_fits=sum(item.get("extra_regime_fits",0) for item in trace))
        write_legacy_evidence_json(destination/"manifest.json",manifest)
    print(f"Estado: {manifest['state']} | Evidencias: {destination}",flush=True)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-run",required=True)
    args = parser.parse_args()
    run_targeted(args.previous_run)


if __name__ == "__main__":
    main()
