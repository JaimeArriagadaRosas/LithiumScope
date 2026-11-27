"""Academic feature ablation, with inner selection and unchanged pilot TEST rows."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from uuid import uuid4
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.paths import RESULTS_DIR, DATA_DIR
from lithiumscope.core.scientific.regression_metrics import regression_metrics
from lithiumscope.datasets.management.sources.local import local_training_source
from lithiumscope.experiments.model_1.balanced.academic import cohort
from lithiumscope.experiments.model_1.expansion.academic_bridge import PILOT, SEED, check_pilot, benchmark_status
from lithiumscope.experiments.model_1.expansion.academic_treatments import fit_profile
from lithiumscope.experiments.model_1.expansion.feature_audit import (
    REFINEMENT, REFINEMENT_SHA, ELEMENTS, derived, numeric_measurement)
from lithiumscope.experiments.model_1.expansion.sensitivity_data import read
from lithiumscope.model_1.preparation.detection_limits import clean_detection_limits
from lithiumscope.model_1.schema import validate_training_schema
from lithiumscope.model_1.training.preprocessing.tabular import FeatureSchema
from lithiumscope.resources.detection.device import detect_device
from lithiumscope.results.atomic_exports import write_table, write_legacy_evidence_json
from lithiumscope.results.reference_benchmarks import OASIS_M1
import re

RECOVERY="20261126T163242_d5f5c5fe"
RECOVERY_SHA="f8469364dc93133a4fe23edcd385844a426f4d19e94ccf950d6e58365ccd7ce8"
BASE="canonical"
ARMS=(BASE,"ratios","measurements","combined")
SELECTED="inner_selected"


def measurement_blocks(frame, recovered, x):
    if not frame.index.equals(x.index) or not recovered.index.equals(x.index):
        raise ValueError("Feature row identities differ")
    fields={}
    for name in frame:
        match=re.fullmatch(r"([A-Za-z]{1,2})_(icpms|xrf)",name)
        if name not in x and match and match[1].lower() in ELEMENTS:
            fields[name]=numeric_measurement(clean_detection_limits(frame[[name]],columns=[name],version=2)[name])
    for name in recovered:
        match=re.fullmatch(r"georoc_([A-Za-z]{1,2})_ppm",name)
        if not match or match[1].lower() not in ELEMENTS or name in fields or name in x:
            raise ValueError("Unexpected recovered measurement: "+name)
        fields[name]=numeric_measurement(recovered[name])
    measured=pd.DataFrame(fields,index=x.index)
    ratios=derived(pd.concat([x,measured],axis=1))
    # Same-source rare earth ratios, without equating unknown methods to ICP-MS.
    for a,b in (("La","Yb"),("La","Sm"),("Gd","Yb"),("Eu","Sm"),("Eu","Gd")):
        for prefix,suffix in (("georoc_","_ppm"),("","_icpms")):
            left,right=prefix+a+suffix,prefix+b+suffix
            if left in measured and right in measured:
                u,v=measured[left],measured[right]
                ratios["log_ratio_"+left+"_"+right]=np.log(u.where(u.gt(0)))-np.log(v.where(v.gt(0)))
    return measured,ratios.replace([np.inf,-np.inf],np.nan)


def select_columns(block, groups):
    if not block.index.equals(groups.index):
        raise ValueError("TRAIN groups misaligned")
    return [name for name in block if block[name].notna().mean()>=.05
            and block[name].nunique()>1 and groups[block[name].notna()].nunique()>=10]


def choose(scores):
    if set(scores.policy)!=set(ARMS) or scores.duplicated(["policy","inner_fold"]).any():
        raise ValueError("Incomplete inner evidence")
    if any(set(g.inner_fold)!={1,2,3} for _,g in scores.groupby("policy")) or not np.isfinite(scores[["rmse","r2"]]).all().all():
        raise ValueError("Invalid inner score")
    agg=scores.groupby("policy")[["rmse","r2"]].mean().reset_index()
    eligible=agg.loc[agg.r2.gt(OASIS_M1["r2_mean"])]
    if eligible.empty:
        return BASE
    return str(eligible.sort_values(["rmse","r2","policy"],ascending=[True,False,True]).iloc[0].policy)


def load_recovery(prepared,digest):
    root=RESULTS_DIR/"generalization/model_1_expansion_linked_recovery"/RECOVERY
    if file_sha256(root/"manifest.json")!=RECOVERY_SHA:
        raise ValueError("Recovery manifest differs")
    m=read(root/"manifest.json")
    if m["source_sha256"]!=digest or m["accepted"]!=2686 or m["training"] is not False:
        raise ValueError("Recovery identity differs")
    for name,sha in m["outputs"].items():
        if file_sha256(root/name)!=sha:
            raise ValueError("Recovery evidence changed: "+name)
    for item in read(root/"raw_inventory.json"):
        name=item["file"]
        from pathlib import Path
        if Path(name).name!=name or file_sha256(DATA_DIR/"raw/model_1/research/georoc_andean_202609"/name)!=item["sha256"]:
            raise ValueError("Raw partition differs")
    r=pd.read_csv(root/"recovered_measurements.csv",dtype={"row_id":str}).set_index("row_id")
    checks=pd.read_csv(root/"row_checks.csv",dtype={"row_id":str})
    if (not r.index.is_unique or set(r.index)!=set(prepared.x.index.astype(str))
            or not checks.row_id.is_unique or not checks.accepted.eq(True).all()
            or len(checks)!=2686 or checks.duplicated(["file","raw_position"]).any()):
        raise ValueError("Recovery row identity differs")
    r=r.reindex(prepared.x.index.astype(str))
    fields=r[m["measurement_columns"]]
    if fields.loc[~fields.index.isin(checks.row_id)].notna().any().any():
        raise ValueError("Measurements outside accepted links")
    if (fields.lt(0)|np.isinf(fields)).any().any() or int(fields.loc[r.academic_member.eq(True)].notna().sum().sum())!=58708:
        raise ValueError("Recovered values differ")
    return r.set_axis(prepared.x.index),m


def run(*, allow_unverified_metadata=False):
    if not allow_unverified_metadata:
        raise ValueError("Explicit --allow-unverified-metadata required; research only")
    source=local_training_source("model_1")
    digest=file_sha256(source)
    prepared,keep,domain=cohort(source)
    x,y,groups,frame=(v.loc[keep] for v in (prepared.x,prepared.y,prepared.groups,prepared.frame))
    if len(x)!=3263 or len(prepared.x)!=3420 or len(x.columns)!=33:
        raise ValueError("Frozen academic cohort differs")
    splits=list(KFold(5,shuffle=True,random_state=SEED).split(x))
    pilot,_=check_pilot(RESULTS_DIR/"generalization/model_1_oasis_like"/PILOT,x,y,groups,splits,digest,domain)
    r,m=load_recovery(prepared,digest)
    if not r.academic_member.eq(keep).all():
        raise ValueError("Academic membership differs")
    measured,ratios=measurement_blocks(frame,r.loc[keep,m["measurement_columns"]],x)
    numeric=list(prepared.schema.numeric)
    categorical=list(prepared.schema.categorical)
    validate_training_schema(prepared.target,numeric+list(measured)+list(ratios),categorical)
    all_features=pd.concat([x,measured,ratios],axis=1)
    reference=RESULTS_DIR/"generalization/model_1_expansion_academic_refinement"/REFINEMENT
    if file_sha256(reference/"manifest.json")!=REFINEMENT_SHA:
        raise ValueError("Previous academic manifest differs")
    rm=read(reference/"manifest.json")
    if file_sha256(reference/"predictions.csv")!=rm["outputs"]["predictions.csv"]:
        raise ValueError("Previous predictions differ")
    ref=pd.read_csv(reference/"predictions.csv",dtype={"row_id":str}).sort_values("row_id").reset_index(drop=True)
    if not ref[["row_id","fold","group"]].equals(pilot[["row_id","fold","group"]]) or not np.allclose(ref.truth,pilot.truth,atol=1e-8,rtol=0):
        raise ValueError("Reference TEST differs")
    out=RESULTS_DIR/"generalization/model_1_expansion_academic_features"/(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")+"_"+uuid4().hex[:8])
    out.mkdir(parents=True,exist_ok=False)
    manifest={"state":"running","scope":"academic_feature_ablation_reused_random_folds",
        "source_sha256":digest,"recovery_manifest_sha256":RECOVERY_SHA,"reference_manifest_sha256":REFINEMENT_SHA,
        "domain":domain,"rows":len(x),"arms":ARMS,"planned_fits":80,"oasis_reference":OASIS_M1,
        "metadata_certified":False,"direct_comparability":False,"independent_holdout":False,
        "retrospective_target_assisted_record_linkage":True,"release_eligible":False,"promotion":False,
        "production_schema_changed":False,"models_saved":False,"new_training_rows":0}
    write_legacy_evidence_json(out/"manifest.json",manifest)
    device=detect_device(prefer_gpu=False)
    predictions,inner_scores,metrics,trace,choices=[],[],[],[],[]
    attempted=0
    def fit(tr,te,policy,fold,stage):
        nonlocal attempted
        extra=[]
        if policy in ("measurements","combined"):
            extra+=select_columns(measured.iloc[tr],groups.iloc[tr])
        if policy in ("ratios","combined"):
            extra+=select_columns(ratios.iloc[tr],groups.iloc[tr])
        schema=FeatureSchema(numeric+extra,categorical,2,"stable_v1")
        cols=schema.numeric+schema.categorical
        tx=all_features.iloc[tr][cols]
        mask=pd.Series(False,index=tx.index)
        attempted+=1
        pred,info=fit_profile(tx,y.iloc[tr],mask,schema,device,SEED,"native_fine")
        trace.append({"fold":fold,"stage":stage,"policy":policy,"features":cols,**info})
        return pred(all_features.iloc[te][cols])
    try:
        for fold,(tr,te) in enumerate(splits,1):
            scores=[]
            for number,(a,b) in enumerate(KFold(3,shuffle=True,random_state=SEED+fold).split(tr),1):
                for policy in ARMS:
                    pred=fit(tr[a],tr[b],policy,fold,f"inner_{number}")
                    scores.append({"fold":fold,"inner_fold":number,"policy":policy,
                                   **regression_metrics(y.iloc[tr[b]],pred).to_dict()})
            inner_scores.extend(scores)
            selected=choose(pd.DataFrame(scores))
            choices.append({"fold":fold,"selected_policy":selected})
            values={p:fit(tr,te,p,fold,"outer") for p in ARMS}
            values[SELECTED]=values[selected]
            for policy,pred in values.items():
                mm=regression_metrics(y.iloc[te],pred).to_dict()
                metrics.append({"fold":fold,"policy":policy,**mm})
                predictions.extend({"fold":fold,"policy":policy,"row_id":str(x.index[i]),
                    "group":str(groups.iloc[i]),"truth":float(y.iloc[i]),"prediction":float(v)}
                    for i,v in zip(te,pred,strict=True))
                print(f"Fold {fold}/5 | {policy} | RMSE={mm['rmse']:.4f} | R2={mm['r2']:.4f}",flush=True)
            write_table(out/"predictions.csv",pd.DataFrame(predictions))
            write_table(out/"inner_scores.csv",pd.DataFrame(inner_scores))
            write_table(out/"fold_metrics.csv",pd.DataFrame(metrics))
            write_legacy_evidence_json(out/"fit_trace.json",trace)
            write_legacy_evidence_json(out/"selection.json",choices)
        report=[]
        for policy,part in pd.DataFrame(predictions).groupby("policy"):
            part=part.sort_values("row_id").reset_index(drop=True)
            if not part.row_id.is_unique or not part[["row_id","fold","group"]].equals(ref[["row_id","fold","group"]]) or not np.allclose(part.truth,ref.truth,atol=1e-8,rtol=0):
                raise ValueError("OOF identity differs")
            fm=pd.DataFrame([regression_metrics(g.truth,g.prediction).to_dict() for _,g in part.groupby("fold")])
            rmse,r2=float(fm.rmse.mean()),float(fm.r2.mean())
            report.append({"policy":policy,"rows":len(part),"rmse_mean":rmse,"r2_mean":r2,"mae_mean":float(fm.mae.mean()),**benchmark_status(rmse,r2)})
        if attempted!=80 or file_sha256(source)!=digest:
            raise ValueError("Campaign incomplete")
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
    parser.add_argument("--allow-unverified-metadata",action="store_true")
    run(allow_unverified_metadata=parser.parse_args().allow_unverified_metadata)
