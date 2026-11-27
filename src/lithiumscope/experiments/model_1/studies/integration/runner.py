"""Phase 4: frozen canonical-input integration, academic and spatial development."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from uuid import uuid4
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GroupKFold

from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.paths import RESULTS_DIR
from lithiumscope.core.reproducibility import runtime_fingerprint
from lithiumscope.core.scientific.regression_metrics import regression_metrics
from lithiumscope.datasets.management.sources.local import local_training_source
from lithiumscope.experiments.data_lab.screening.audit import REFERENCE,FOLD_SHA256
from lithiumscope.experiments.model_1.protocols.partitions import outer_from_manifest,validate_partition
from lithiumscope.experiments.model_1.protocols.cohorts import cohort
from lithiumscope.experiments.model_1.studies.expansion.academic_bridge import PILOT,SEED,check_pilot,benchmark_status
from lithiumscope.experiments.model_1.training_methods.boosting import archived_run,archive_predictions,PROFILES_NEW
from lithiumscope.experiments.model_1.training_methods.losses import loss_params,score,summarize_scores
from lithiumscope.experiments.model_1.evaluation.inference_contract import POLICIES,canonical_columns,fit_bases,predict_bases,fit_ridge,paired_crossfit
from lithiumscope.experiments.model_1.data_expansion.sensitivity_data import read
from lithiumscope.model_1.preparation.prediction_data import prepare_prediction_data
from lithiumscope.model_1.schema import validate_training_schema
from lithiumscope.resources.detection.device import detect_device
from lithiumscope.results.atomic_exports import write_table,write_legacy_evidence_json

SOURCE_SHA="94e78b2662b002184bcfd550be0aacc522f5eec73283d0c12f75be29599bcc94"
STACK_RUN="20261126T231135_4ec86481"
STACK_SHA="bc54c943eb5a1c350c2104d0b3df9174da9811a6360324b01430e9a4c8804350"
RF_RUN="training_20261125_115932_m0300_826123c1"
MAX_BASE_FITS=80


def check_inference_parity(prepared):
    cols=canonical_columns(prepared.schema)
    validate_training_schema(prepared.target,prepared.schema.numeric,prepared.schema.categorical)
    raw=prepared.raw_frame.loc[prepared.x.index].drop(columns=[prepared.target])
    predicted=prepare_prediction_data(raw,"common",prepared.schema)
    pd.testing.assert_frame_equal(predicted[cols],prepared.x[cols],check_dtype=False,
                                  check_exact=False,rtol=1e-10,atol=1e-10)
    return {"rows":len(predicted),"columns":cols,"target_removed":True,"parity":True}


def spatial_plan(x,groups):
    path=RESULTS_DIR/"model_1/runs"/REFERENCE/"tables/outer_fold_manifest.csv"
    if file_sha256(path)!=FOLD_SHA256:
        raise ValueError("Frozen spatial fold manifest differs")
    f=pd.read_csv(path,dtype={"row_index":str}).sort_values("row_position")
    if (not np.array_equal(f.row_position,np.arange(len(x)))
            or not np.array_equal(f.row_index,x.index.astype(str))
            or not np.array_equal(f.spatial_group,groups.astype(str))
            or not np.allclose(f[["Longitude","Latitude"]],x[["Longitude","Latitude"]],rtol=0,atol=1e-8)):
        raise ValueError("Frozen spatial identity differs")
    return outer_from_manifest(f.outer_fold,groups,5),f.outer_fold.to_numpy(int)


def align_spatial_reference(pred, manifest, x, y, groups, fold_ids):
    """Recover legacy OOF identity from its own manifest, never from row order alone."""
    identity=["row_position","row_index","outer_fold","spatial_group","Longitude","Latitude"]
    required={"row_index","Li_icpms_real","Li_icpms_predicted"}
    if not required.issubset(pred.columns) or not set(identity).issubset(manifest.columns):
        raise ValueError("Missing historical OOF identity fields")
    if (len(pred)!=len(x) or len(manifest)!=len(x) or not x.index.is_unique
            or pred.row_index.isna().any() or manifest[identity].isna().any().any()
            or pred.row_index.duplicated().any() or manifest.row_index.duplicated().any()):
        raise ValueError("Invalid historical OOF identities")
    m=manifest.sort_values("row_position").reset_index(drop=True)
    if (not np.array_equal(m.row_position,np.arange(len(x)))
            or not np.array_equal(m.row_index,x.index.astype(str))
            or not np.array_equal(m.outer_fold,fold_ids)
            or not np.array_equal(m.spatial_group,groups.astype(str))
            or not np.allclose(m[["Longitude","Latitude"]],x[["Longitude","Latitude"]],rtol=0,atol=1e-8)
            or set(pred.row_index)!=set(m.row_index)):
        raise ValueError("Historical OOF manifest identity differs")
    p=pred.set_index("row_index").reindex(m.row_index)
    # Modern exports may repeat identity fields. Verify rather than overwrite.
    for key in identity:
        if key=="row_index" or key not in p:
            continue
        a,b=p[key].to_numpy(),m[key].to_numpy()
        equal=np.allclose(a,b,rtol=0,atol=1e-8) if key in ("Longitude","Latitude") else np.array_equal(a,b)
        if not equal:
            raise ValueError("Historical OOF redundant identity differs: "+key)
    if (not np.allclose(p.Li_icpms_real,y,rtol=0,atol=1e-8)
            or not np.isfinite(p.Li_icpms_predicted).all()):
        raise ValueError("Historical OOF truth or prediction differs")
    return p.Li_icpms_predicted.to_numpy(float)


def spatial_reference(run,algorithm,x,y,groups,fold_ids,digest,expected):
    root=RESULTS_DIR/"model_1/runs"/run
    record=read(root/"run.json")
    if record["state"]!="completed" or record["summary"]["dataset_sha256"]!=digest:
        raise ValueError("Spatial reference source differs")
    path=root/"tables"/f"oof_predictions_{algorithm}__identity.csv"
    manifest_path=root/"tables"/"outer_fold_manifest.csv"
    hashes={"oof_sha256":file_sha256(path),"fold_manifest_sha256":file_sha256(manifest_path)}
    pred=pd.read_csv(path,dtype={"row_index":str})
    manifest=pd.read_csv(manifest_path,dtype={"row_index":str})
    p=align_spatial_reference(pred,manifest,x,y,groups,fold_ids)
    rmse=regression_metrics(y,p).rmse
    if not np.isclose(rmse,expected,rtol=0,atol=1e-5):
        raise ValueError("Spatial reference RMSE differs")
    if hashes!={"oof_sha256":file_sha256(path),"fold_manifest_sha256":file_sha256(manifest_path)}:
        raise ValueError("Spatial reference changed during read")
    return p,hashes


def stats(part):
    fm=pd.DataFrame([score(g.truth,g.prediction) for _,g in part.groupby("fold")])
    agg=summarize_scores(fm)
    gm=regression_metrics(part.truth,part.prediction)
    ext=part.loc[part.truth.gt(696)]
    return {**agg,"rmse_global":gm.rmse,"r2_global":gm.r2,"mae_global":gm.mae,
            "extreme_n":len(ext),"extreme_rmse":float(np.sqrt(np.mean((ext.truth-ext.prediction)**2))) if len(ext) else None,
            "tail_bias":float((part.loc[part.truth.gt(40),"prediction"]-part.loc[part.truth.gt(40),"truth"]).mean())}


def better(row,base,*,spatial=False,strict=True):
    # Predeclared development guards, not production eligibility or scientific proof.
    op=(lambda a,b:a<b) if strict else (lambda a,b:a<=b+1e-10)
    return bool(op(row["rmse_mean"],base["rmse_mean"]) and op(row["rmse_global"],base["rmse_global"])
        and row["tail_rmse"]<=base["tail_rmse"]+1e-10 and row["mae_global"]<=base["mae_global"]+.1
        and (row["extreme_n"]==base["extreme_n"] and row["extreme_rmse"]<=base["extreme_rmse"]+1e-10 if spatial else row["r2_mean"]>.499))


def decision(report):
    by={(r["protocol"],r["policy"]):r for r in report}
    h="hgb_canonical"
    a=by["academic",h]
    s=by["spatial",h]
    result=[]
    for name in (h,"ridge_canonical"):
        ar,sr=by["academic",name],by["spatial",name]
        internal=(ar["r2_mean"]>.499 if name==h else better(ar,a) and better(sr,s,spatial=True))
        refs=all(better(sr,by["spatial",ref],spatial=True) for ref in ("archived_svm","archived_rf"))
        result.append({"policy":name,"canonical_guard":bool(internal),"spatial_reference_guard":bool(refs),
                       "reference_review_eligible":bool(internal and refs)})
    return result


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
    spatial,fold_ids=spatial_plan(x,g)
    svm,svm_sha=spatial_reference(REFERENCE,"svm",x,y,g,fold_ids,digest,159.846300)
    rf,rf_sha=spatial_reference(RF_RUN,"random_forest",x,y,g,fold_ids,digest,152.302214)
    ax,ay,ag=(v.loc[keep] for v in (x,y,g))
    academic=list(KFold(5,shuffle=True,random_state=SEED).split(ax))
    pilot,_=check_pilot(RESULTS_DIR/"generalization/model_1_oasis_like"/PILOT,ax,ay,ag,academic,digest,domain)
    stack,_=archived_run("model_1_integral_stacking",STACK_RUN,STACK_SHA,digest)
    ah=archive_predictions(stack,"identity",pilot,ax)
    ar=archive_predictions(stack,"ridge_stack",pilot,ax)
    out=RESULTS_DIR/"generalization/model_1_integral_integration"/(
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")+"_"+uuid4().hex[:8])
    out.mkdir(parents=True,exist_ok=False)
    manifest={"state":"running","scope":"canonical_input_integrated_development","source_sha256":digest,
        "runtime":runtime_fingerprint(),"inference_parity":parity,"domain":domain,
        "spatial_fold_sha256":FOLD_SHA256,"svm_oof_sha256":svm_sha["oof_sha256"],"rf_oof_sha256":rf_sha["oof_sha256"],
        "svm_fold_manifest_sha256":svm_sha["fold_manifest_sha256"],"rf_fold_manifest_sha256":rf_sha["fold_manifest_sha256"],
        "stack_manifest_sha256":STACK_SHA,"profiles":{"hgb":loss_params("squared_error"),"xgb":PROFILES_NEW["xgb_regularized"][1]},
        "max_base_fits":MAX_BASE_FITS,"planned_ridge_fits":10,"reused_development_folds":True,
        "recovered_measurements_used":False,"target_assisted_linkage_used":False,"independent_holdout":False,
        "direct_comparability":False,"production_schema_changed":False,"new_training_rows":0,
        "models_saved":False,"promotion":False,"release_eligible":False}
    write_legacy_evidence_json(out/"manifest.json",manifest)
    device=detect_device(prefer_gpu=False)
    fits=0
    rows,trace,calibration,fold_metrics,report=[],[],[],[],[]
    try:
        for protocol,px,py,pg,splits,refs in (
            ("spatial",x,y,g,spatial,{"archived_svm":svm,"archived_rf":rf}),
            ("academic",ax,ay,ag,academic,{"enriched_hgb":ah.prediction.to_numpy(float),"enriched_ridge":ar.prediction.to_numpy(float)})):
            for fold,(tr,te) in enumerate(splits,1):
                if protocol=="spatial":
                    inner=[validate_partition(tr[a],tr[b],tr,pg) for a,b in GroupKFold(3).split(tr,groups=pg.iloc[tr])]
                else:
                    inner=[(tr[a],tr[b]) for a,b in KFold(3,shuffle=True,random_state=SEED+fold).split(tr)]
                def fit_predict(a,b):
                    nonlocal fits
                    if set(a)&set(b) or (protocol=="spatial" and set(pg.iloc[a])&set(pg.iloc[b])):
                        raise ValueError("Training evaluation overlap")
                    models=fit_bases(px.iloc[a],py.iloc[a],prepared.schema,device)
                    fits+=2
                    trace.append({"protocol":protocol,"fold":fold,"train_ids":px.index[a].astype(str).tolist(),
                        "evaluation_ids":px.index[b].astype(str).tolist()})
                    return predict_bases(models,px.iloc[b],prepared.schema)
                hc,xc=paired_crossfit(tr,inner,fit_predict)
                correction=fit_ridge(py.iloc[tr],hc,xc)
                calibration.extend({"protocol":protocol,"fold":fold,"row_id":str(px.index[i]),"truth":float(py.iloc[i]),
                    "hgb_oof":float(h),"xgb_oof":float(z)} for i,h,z in zip(tr,hc,xc,strict=True))
                hp,xp=fit_predict(tr,te)
                values=dict(zip(POLICIES,(hp,xp,correction.predict(hp,xp)),strict=True))
                values.update({name:p[te] for name,p in refs.items()})
                for name,p in values.items():
                    mm=score(py.iloc[te],p)
                    fold_metrics.append({"protocol":protocol,"fold":fold,"policy":name,**mm})
                    rows.extend({"protocol":protocol,"fold":fold,"policy":name,"row_id":str(px.index[i]),
                        "group":str(pg.iloc[i]),"truth":float(py.iloc[i]),"prediction":float(v)}
                        for i,v in zip(te,p,strict=True))
                    print(f"{protocol} fold {fold}/5 | {name} | RMSE={mm['rmse']:.4f} | R2={mm['r2']:.4f}",flush=True)
                write_table(out/"predictions.csv",pd.DataFrame(rows))
                write_table(out/"fold_metrics.csv",pd.DataFrame(fold_metrics))
                write_table(out/"calibration_oof.csv",pd.DataFrame(calibration))
                write_legacy_evidence_json(out/"fit_trace.json",trace)
        table=pd.DataFrame(rows)
        for (protocol,name),part in table.groupby(["protocol","policy"]):
            ref=table.loc[table.protocol.eq(protocol)&table.policy.eq("hgb_canonical")].sort_values("row_id").reset_index(drop=True)
            part=part.sort_values("row_id").reset_index(drop=True)
            keys=["row_id","fold","group","truth"]
            if not part.row_id.is_unique or len(part)!=(3420 if protocol=="spatial" else 3263) or not part[keys].equals(ref[keys]):
                raise ValueError("Integration OOF identity differs")
            agg=stats(part)
            report.append({"protocol":protocol,"policy":name,**agg,
                **(benchmark_status(agg["rmse_mean"],agg["r2_mean"]) if protocol=="academic" else {})})
        if fits!=MAX_BASE_FITS or len(trace)!=40 or len(report)!=10 or file_sha256(source)!=digest:
            raise ValueError("Integration incomplete")
        decisions=decision(report)
        write_table(out/"summary.csv",pd.DataFrame(report))
        write_legacy_evidence_json(out/"decision.json",decisions)
        manifest["summary"]=report
        manifest["decision"]=decisions
        manifest["state"]="complete"
    except BaseException as exc:
        manifest["state"]="interrupted"
        manifest["error"]=str(exc)
        raise
    finally:
        manifest["base_fits"]=fits
        manifest["outputs"]={p.name:file_sha256(p) for p in out.iterdir() if p.is_file() and p.name!="manifest.json"}
        write_legacy_evidence_json(out/"manifest.json",manifest)
    print(pd.DataFrame(report).to_string(index=False))
    print(pd.DataFrame(decisions).to_string(index=False))
    print(f"Estado: {manifest['state']} | Evidencias: {out}",flush=True)
    return out


if __name__=="__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    run()
