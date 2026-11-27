"""Fixed academic benchmark on the unchanged 3263-row OASIS-like pilot cohort."""
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
from lithiumscope.datasets.management.sources.local import local_training_source
from lithiumscope.experiments.model_1.balanced.academic import cohort
from lithiumscope.experiments.model_1.expansion.sensitivity_data import load_evidence, historical_tokens, reserve, fold_pools, read
from lithiumscope.experiments.model_1.targeted.fit import _base_fit
from lithiumscope.experiments.model_1.targeted.policies import Arm
from lithiumscope.resources.detection.device import detect_device
from lithiumscope.results.atomic_exports import write_table, write_legacy_evidence_json
from lithiumscope.results.reference_benchmarks import OASIS_M1

PILOT="20261126T123613_24ef3f6c"
SEED=42
# Fixed before the academic campaign; no use of archived spatial-fold
# hyperparameters, which may have been fitted using these random TEST rows.
PROFILES={
    "random_forest":{"n_estimators":500,"min_samples_leaf":1,"max_features":.8,"treatment_policy":"nonnegative"},
    "catboost":{"iterations":700,"depth":6,"learning_rate":.05,"l2_leaf_reg":3.,"treatment_policy":"nonnegative"},
    "hist_gradient_boosting":{"max_iter":500,"max_leaf_nodes":31,"min_samples_leaf":10,
                             "learning_rate":.05,"l2_regularization":.1,"treatment_policy":"nonnegative"},
}
ARMS=("historical","compatible_expansion")


def benchmark_status(rmse_mean,r2_mean):
    rmse=float(rmse_mean)
    r2=float(r2_mean)
    return {"rmse_better":bool(np.isfinite(rmse) and rmse<OASIS_M1["rmse_mean"]),
            "r2_better":bool(np.isfinite(r2) and r2>OASIS_M1["r2_mean"]),
            "both_better":bool(np.isfinite(rmse) and np.isfinite(r2)
                               and rmse<OASIS_M1["rmse_mean"] and r2>OASIS_M1["r2_mean"])}


def check_pilot(pilot,x,y,groups,splits,digest,domain):
    manifest=read(pilot/"manifest.json")
    if (manifest.get("state")!="complete" or manifest.get("source_sha256")!=digest
            or manifest.get("rows")!=3263 or manifest.get("seed")!=SEED
            or manifest.get("outer_folds")!=5 or manifest.get("domain")!=domain):
        raise ValueError("Frozen academic pilot manifest differs")
    previous=pd.read_csv(pilot/"predictions.csv",dtype={"row_id":str})
    expected=[]
    for fold,(_,te) in enumerate(splits,1):
        expected.extend({"row_id":str(x.index[i]),"fold":fold,"group":str(groups.iloc[i]),
                         "truth":float(y.iloc[i])} for i in te)
    expected=pd.DataFrame(expected).sort_values("row_id").reset_index(drop=True)
    previous=previous.sort_values("row_id").reset_index(drop=True)
    keys=["row_id","fold","group"]
    if (not previous.row_id.is_unique or len(previous)!=len(expected)
            or not previous[keys].equals(expected[keys])
            or not np.allclose(previous.truth,expected.truth,rtol=0,atol=1e-8)
            or not np.isfinite(previous.prediction).all()):
        raise ValueError("Academic pilot TEST identities differ")
    fold_metrics=[regression_metrics(g.truth,g.prediction) for _,g in previous.groupby("fold")]
    rmse=float(np.mean([r.rmse for r in fold_metrics]))
    r2=float(np.mean([r.r2 for r in fold_metrics]))
    if not np.allclose([rmse,r2],[7.723715653,.561373693],rtol=0,atol=1e-8):
        raise ValueError("Archived academic pilot metrics differ")
    return previous,{"rmse_mean":rmse,"r2_mean":r2,**benchmark_status(rmse,r2)}


def range_pool(pool,targets,domain):
    """Training candidates must lie within the already frozen academic domain."""
    y=np.asarray(targets,float)
    mask=np.asarray(pool,bool)
    if mask.shape!=y.shape or not np.isfinite(y).all():
        raise ValueError("Invalid academic candidate labels")
    return mask & (y>=domain["q025"]) & (y<=domain["q975"])


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
        raise ValueError("Academic historical cohort or schema changed")
    x,y,groups=(v.loc[keep] for v in (prepared.x,prepared.y,prepared.groups))
    numeric=list(prepared.schema.numeric)
    categorical=list(prepared.schema.categorical)
    if numeric!=evidence["numeric"] or categorical!=evidence["categorical"]:
        raise ValueError("Academic predictor schema changed")
    all_refs=historical_tokens(prepared.frame)
    refs=[all_refs[i] for i in np.flatnonzero(keep.to_numpy())]
    # Preserve the SAME reserve and quarantine as the full-domain campaigns.
    held,pool=reserve(audit,pubs,prepared.groups,all_refs)
    cx=features[list(x.columns)].copy()
    for key in numeric:
        cx[key]=pd.to_numeric(cx[key],errors="raise").astype(float)
    for key in categorical:
        cx[key]=cx[key].astype(object).where(cx[key].notna(),np.nan)
    cy=pd.to_numeric(features.Li_icpms,errors="raise")
    cg=features.spatial_group.astype(str)
    cs=features.source_dataset.astype(str)
    pool=range_pool(pool,cy,domain)&~audit.screening_state.eq("high_transfer_review").to_numpy()
    # Historical source label is diagnostic, not a predictor.
    from lithiumscope.core.scientific.eda_statistics import source_series
    sources=source_series(prepared.frame).loc[keep]
    splits=list(KFold(5,shuffle=True,random_state=SEED).split(x))
    pilot=RESULTS_DIR/"generalization/model_1_oasis_like"/PILOT
    previous,pilot_metrics=check_pilot(pilot,x,y,groups,splits,digest,domain)
    device=detect_device(prefer_gpu=False)
    out=RESULTS_DIR/"generalization/model_1_expansion_academic"/(
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")+"_"+uuid4().hex[:8])
    out.mkdir(parents=True,exist_ok=False)
    manifest={**links,"state":"running","scope":"academic_fixed_profiles_frozen_pilot_population",
        "source_sha256":digest,"pilot_id":PILOT,"pilot_manifest_sha256":file_sha256(pilot/"manifest.json"),
        "pilot_predictions_sha256":file_sha256(pilot/"predictions.csv"),"pilot_metrics":pilot_metrics,
        "domain":domain,"rows":len(x),"seed":SEED,"outer_folds":5,"profiles":PROFILES,"arms":list(ARMS),
        "planned_fits":30,"oasis_reference":OASIS_M1,"direct_comparability":False,
        "random_cv_spatial_overlap":True,"nested_tuning":False,"metadata_certified":False,
        "independent_holdout":False,"release_eligible":False,"promotion":False,"models_saved":False,
        "reserved_rows_excluded":int(held.sum()),"in_range_candidate_pool":int(pool.sum())}
    write_legacy_evidence_json(out/"manifest.json",manifest)
    predictions,metrics,plans,trace=[],[],[],[]
    attempted=0
    try:
        for fold,(tr,te) in enumerate(splits,1):
            blocked=set().union(*(refs[i] for i in te))
            pos=fold_pools(audit,cx,x.iloc[tr],numeric,pool,pubs,set(groups.iloc[te]),blocked)["D4"]
            if not len(pos):
                raise ValueError("No compatible new candidates in academic TRAIN")
            if held[pos].any() or set(cg.iloc[pos])&set(groups.iloc[te]) or any(pubs[i]&blocked for i in pos):
                raise ValueError("Candidate expansion leaks into academic TEST or reserve")
            plans.append({"fold":fold,"historical_train_ids":x.index[tr].astype(str).tolist(),
                "historical_test_ids":x.index[te].astype(str).tolist(),
                "candidate_train_ids":audit.candidate_id.iloc[pos].tolist()})
            for algorithm,params in PROFILES.items():
                for arm in ARMS:
                    selected=pos if arm=="compatible_expansion" else np.array([],dtype=int)
                    bx,by,bg,bs=[v.iloc[tr].copy() for v in (x,y,groups,sources)]
                    if len(selected):
                        ids=pd.Index("candidate:"+audit.candidate_id.iloc[selected].astype(str))
                        added=[v.iloc[selected].set_axis(ids) for v in (cx,cy,cg,cs)]
                        bx,by,bg,bs=[pd.concat([a,b]) for a,b in zip((bx,by,bg,bs),added,strict=True)]
                    if not bx.index.is_unique:
                        raise ValueError("Duplicate fit identity")
                    attempted+=1
                    predictor,info=_base_fit(algorithm,Arm("control"),device,SEED,params,bx,by,bg,bs,numeric,categorical)
                    score=np.maximum(np.asarray(predictor(x.iloc[te]),float),0.)
                    values=regression_metrics(y.iloc[te],score).to_dict()
                    metrics.append({"algorithm":algorithm,"policy":arm,"fold":fold,
                        "historical_train_rows":len(tr),"candidate_train_rows":len(selected),
                        "groups_shared_train_test":len(set(groups.iloc[tr])&set(groups.iloc[te])),**values})
                    predictions.extend({"algorithm":algorithm,"policy":arm,"fold":fold,
                        "row_id":str(x.index[i]),"group":str(groups.iloc[i]),"truth":float(y.iloc[i]),
                        "prediction":float(v)} for i,v in zip(te,score,strict=True))
                    trace.append({"algorithm":algorithm,"policy":arm,"fold":fold,**info})
                    print(f"Academic {fold}/5 | {algorithm} | {arm} | new={len(selected)} | RMSE={values['rmse']:.4f} | R2={values['r2']:.4f}",flush=True)
            write_table(out/"predictions.csv",pd.DataFrame(predictions))
            write_table(out/"fold_metrics.csv",pd.DataFrame(metrics))
            write_legacy_evidence_json(out/"partitions.json",plans)
            write_legacy_evidence_json(out/"fit_trace.json",trace)
        pred=pd.DataFrame(predictions)
        fm=pd.DataFrame(metrics)
        report=[]
        for (algorithm,arm),part in pred.groupby(["algorithm","policy"]):
            part=part.sort_values("row_id").reset_index(drop=True)
            if not part[["row_id","fold","group","truth"]].equals(previous[["row_id","fold","group","truth"]]):
                # CSV round-trip can differ at the last floating point bit.
                if (not part[["row_id","fold","group"]].equals(previous[["row_id","fold","group"]])
                        or not np.allclose(part.truth,previous.truth,rtol=0,atol=1e-8)):
                    raise ValueError("Academic OOF mismatch")
            f=fm[(fm.algorithm==algorithm)&(fm.policy==arm)]
            rmse,r2=float(f.rmse.mean()),float(f.r2.mean())
            report.append({"algorithm":algorithm,"policy":arm,"rows":len(part),
                "rmse_mean":rmse,"rmse_std":float(f.rmse.std(ddof=1)),
                "r2_mean":r2,"r2_std":float(f.r2.std(ddof=1)),"mae_mean":float(f.mae.mean()),
                **{"global_"+k:v for k,v in regression_metrics(part.truth,part.prediction).to_dict().items()},
                **benchmark_status(rmse,r2)})
        if len(report)!=6 or attempted!=30 or file_sha256(source)!=digest:
            raise ValueError("Academic campaign incomplete")
        write_table(out/"summary.csv",pd.DataFrame(report))
        manifest["summary"]=report
        manifest["state"]="complete"
    except BaseException as exc:
        manifest["state"]="interrupted"
        manifest["error"]=str(exc)
        raise
    finally:
        manifest["attempted_fits"]=attempted
        manifest["outputs"]={p.name:file_sha256(p) for p in out.iterdir() if p.is_file() and p.name!="manifest.json"}
        write_legacy_evidence_json(out/"manifest.json",manifest)
    print(pd.DataFrame(report).to_string(index=False))
    print(f"Estado: {manifest['state']} | Evidencias: {out}",flush=True)
    return out


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",required=True)
    parser.add_argument("--allow-unverified-metadata",action="store_true")
    args=parser.parse_args()
    run(args.run,allow_unverified_metadata=args.allow_unverified_metadata)
