"""Phase 4 recovery: fixed RF fallback on verified canonical integration OOF."""
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
from lithiumscope.datasets.management.local_sources import local_training_source
from lithiumscope.experiments.model_1.balanced.academic import cohort
from lithiumscope.experiments.model_1.expansion.academic_bridge import PROFILES,SEED,benchmark_status
from lithiumscope.experiments.model_1.expansion.sensitivity_data import read
from lithiumscope.experiments.model_1.integral.integration import SOURCE_SHA,spatial_plan,check_inference_parity,stats,better
from lithiumscope.experiments.model_1.integral.spatial_support import POLICIES,PRIMARY,support,apply_policies
from lithiumscope.model_1.training.factory import build_pipeline
from lithiumscope.resources.detection.device import detect_device
from lithiumscope.resources.manager import estimator_thread_scope
from lithiumscope.results.atomic_exports import write_table,write_legacy_evidence_json

RUN="20261126T232226_c7612b4d"
SHA="0ec679dfe7c15a318869bbcdb65bbb5e148dfba089ade5d993da5649db9b5d72"
RF_PARAMS={k:v for k,v in PROFILES["random_forest"].items() if k!="treatment_policy"}


def load_integration(digest):
    root=RESULTS_DIR/"generalization/model_1_integral_integration"/RUN
    if file_sha256(root/"manifest.json")!=SHA:
        raise ValueError("Integration manifest differs")
    m=read(root/"manifest.json")
    if m["state"]!="complete" or m["source_sha256"]!=digest or m["target_assisted_linkage_used"]:
        raise ValueError("Integration scope differs")
    for name,h in m["outputs"].items():
        if file_sha256(root/name)!=h:
            raise ValueError("Integration output differs: "+name)
    p=pd.read_csv(root/"predictions.csv",dtype={"row_id":str})
    if len(p)!=33415:
        raise ValueError("Incomplete archived integration")
    return p


def aligned(table,protocol,policy,fold,x,y,g,te):
    part=table.loc[table.protocol.eq(protocol)&table.policy.eq(policy)&table.fold.eq(fold)]
    ids=x.index[te].astype(str)
    if not part.row_id.is_unique or set(part.row_id)!=set(ids):
        raise ValueError("Archived integration row identities differ")
    part=part.set_index("row_id").reindex(ids)
    if (not np.allclose(part.truth,y.iloc[te],rtol=0,atol=1e-8)
            or not np.array_equal(part.group,g.iloc[te].astype(str))
            or not np.isfinite(part.prediction).all()):
        raise ValueError("Archived integration truth/group differs")
    return part.prediction.to_numpy(float)


def fit_rf(x,y,schema,device):
    if not x.index.is_unique or not x.index.equals(y.index) or not np.isfinite(y).all() or y.lt(0).any():
        raise ValueError("Invalid RF TRAIN")
    model=build_pipeline("random_forest",device,SEED,schema,RF_PARAMS)
    with estimator_thread_scope():
        model.fit(x,y)
    return model


def decisions(report):
    by={(r["protocol"],r["policy"]):r for r in report}
    # Primary was specified before execution. Other gates are diagnostics only.
    a,s=by["academic",PRIMARY],by["spatial",PRIMARY]
    academic=better(a,by["academic","hgb_canonical"])
    spatial=all(better(s,by["spatial",r],spatial=True)
                for r in ("archived_svm","archived_rf","hgb_canonical"))
    return {"primary_policy":PRIMARY,"academic_guard":academic,"spatial_guard":spatial,
            "reference_review_eligible":bool(academic and spatial)}


def run():
    source=local_training_source("model_1")
    digest=file_sha256(source)
    if digest!=SOURCE_SHA:
        raise ValueError("Frozen source differs")
    prepared,keep,domain=cohort(source)
    x,y,g=prepared.x,prepared.y,prepared.groups
    if len(x)!=3420 or g.nunique()!=244 or int(keep.sum())!=3263:
        raise ValueError("Frozen cohort differs")
    parity=check_inference_parity(prepared)
    old=load_integration(digest)
    spatial,_=spatial_plan(x,g)
    ax,ay,ag=(v.loc[keep] for v in (x,y,g))
    academic=list(KFold(5,shuffle=True,random_state=SEED).split(ax))
    out=RESULTS_DIR/"generalization/model_1_integral_spatial_recovery"/(
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")+"_"+uuid4().hex[:8])
    out.mkdir(parents=True,exist_ok=False)
    manifest={"state":"running","scope":"retrospective_fixed_rf_fallback_development",
        "source_sha256":digest,"integration_manifest_sha256":SHA,"runtime":runtime_fingerprint(),
        "domain":domain,"inference_parity":parity,"primary_policy":PRIMARY,
        "policies":POLICIES,"rf_params":RF_PARAMS,"planned_fits":10,
        "thresholds":{"prediction_ppm":40,"relative_disagreement":2,"absolute_disagreement_ppm":20,
                      "neighbors":15,"high_support_group_min":2,"radius_multiplier":2,"label_fraction":.5},
        "reused_development_folds":True,"target_assisted_linkage_used":False,
        "independent_holdout":False,"new_training_rows":0,"models_saved":False,
        "promotion":False,"release_eligible":False}
    write_legacy_evidence_json(out/"manifest.json",manifest)
    device=detect_device(prefer_gpu=False)
    rows,diags,traces,report=[],[],[],[]
    fits=0
    try:
        for protocol,px,py,pg,splits in (("spatial",x,y,g,spatial),("academic",ax,ay,ag,academic)):
            refs=("hgb_canonical","ridge_canonical","archived_svm","archived_rf") if protocol=="spatial" else ("hgb_canonical","ridge_canonical")
            for fold,(tr,te) in enumerate(splits,1):
                if set(tr)&set(te) or (protocol=="spatial" and set(pg.iloc[tr])&set(pg.iloc[te])):
                    raise ValueError("TRAIN evaluation overlap")
                values={p:aligned(old,protocol,p,fold,px,py,pg,te) for p in refs}
                rf=fit_rf(px.iloc[tr],py.iloc[tr],prepared.schema,device)
                fits+=1
                with estimator_thread_scope():
                    rp=np.maximum(np.asarray(rf.predict(px.iloc[te]),float),0.)
                ridge=values["ridge_canonical"]
                diag=support(px.iloc[tr],py.iloc[tr],pg.iloc[tr],px.iloc[te],ridge)
                corrected,flags=apply_policies(ridge,rp,diag)
                values.update({"rf_fixed":rp,**corrected})
                traces.append({"protocol":protocol,"fold":fold,"train_ids":px.index[tr].astype(str).tolist(),
                               "evaluation_ids":px.index[te].astype(str).tolist()})
                diag=diag.reset_index(drop=True)
                diag["protocol"]=protocol
                diag["fold"]=fold
                diag["row_id"]=px.index[te].astype(str)
                diag["ridge_prediction"]=ridge
                diag["rf_prediction"]=rp
                for name,mask in flags.items():
                    diag[name+"_fallback"]=mask
                diags.extend(diag.to_dict("records"))
                for name,p in values.items():
                    rows.extend({"protocol":protocol,"fold":fold,"policy":name,"row_id":str(px.index[i]),
                                 "group":str(pg.iloc[i]),"truth":float(py.iloc[i]),"prediction":float(v)}
                                for i,v in zip(te,p,strict=True))
                print(f"{protocol} fold {fold}/5 | primary fallback={int(flags[PRIMARY].sum())}/{len(te)}",flush=True)
                write_table(out/"predictions.csv",pd.DataFrame(rows))
                write_table(out/"support.csv",pd.DataFrame(diags))
                write_legacy_evidence_json(out/"fit_trace.json",traces)
        table=pd.DataFrame(rows)
        for (protocol,name),part in table.groupby(["protocol","policy"]):
            ref=table.loc[table.protocol.eq(protocol)&table.policy.eq("hgb_canonical")].sort_values("row_id").reset_index(drop=True)
            part=part.sort_values("row_id").reset_index(drop=True)
            keys=["row_id","fold","group","truth"]
            if not part.row_id.is_unique or not part[keys].equals(ref[keys]):
                raise ValueError("Recovery OOF identities differ")
            a=stats(part)
            report.append({"protocol":protocol,"policy":name,**a,
                           **(benchmark_status(a["rmse_mean"],a["r2_mean"]) if protocol=="academic" else {})})
        if fits!=10 or len(diags)!=6683 or len(rows)!=46938 or file_sha256(source)!=digest:
            raise ValueError("Recovery incomplete")
        result=decisions(report)
        write_table(out/"summary.csv",pd.DataFrame(report))
        write_legacy_evidence_json(out/"decision.json",result)
        manifest.update(state="complete",summary=report,decision=result)
    except BaseException as exc:
        manifest.update(state="interrupted",error=str(exc))
        raise
    finally:
        manifest["fits"]=fits
        manifest["outputs"]={p.name:file_sha256(p) for p in out.iterdir() if p.is_file() and p.name!="manifest.json"}
        write_legacy_evidence_json(out/"manifest.json",manifest)
    print(pd.DataFrame(report).to_string(index=False))
    print(result)
    print(f"Estado: {manifest['state']} | Evidencias: {out}",flush=True)
    return out


if __name__=="__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    run()
