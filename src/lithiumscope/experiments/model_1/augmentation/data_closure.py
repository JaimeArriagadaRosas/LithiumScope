"""One bounded academic data-composition closure, never a release or independent test."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from uuid import uuid4
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.paths import RESULTS_DIR
from lithiumscope.core.scientific.eda_statistics import source_series
from lithiumscope.core.scientific.regression_metrics import regression_metrics
from lithiumscope.datasets.management.sources.local import local_training_source
from lithiumscope.experiments.model_1.balanced.academic import cohort
from lithiumscope.experiments.model_1.expansion.academic_bridge import PILOT, SEED, PROFILES, check_pilot, benchmark_status
from lithiumscope.experiments.model_1.expansion.academic_final import load_reference, FEATURE_SHA
from lithiumscope.experiments.model_1.expansion.academic_features import load_recovery, measurement_blocks, select_columns, RECOVERY_SHA
from lithiumscope.experiments.model_1.expansion.sensitivity_data import read
from lithiumscope.experiments.model_1.augmentation.transfer_preflight import CHEN_RUN, CHEN_MANIFEST, verified_run
from lithiumscope.experiments.model_1.augmentation.chen_profile import ROOT, SOURCE, NAME, SHA
from lithiumscope.experiments.model_1.augmentation.treatments import augment
from lithiumscope.model_1.preparation.feature_engineering import add_geochemical_features
from lithiumscope.model_1.training.preprocessing.tabular import FeatureSchema
from lithiumscope.model_1.schema import validate_training_schema
from lithiumscope.model_1.training.factory import build_pipeline
from lithiumscope.resources.detection.device import detect_device
from lithiumscope.resources.manager import estimator_thread_scope
from lithiumscope.results.atomic_exports import write_table, write_legacy_evidence_json

PREFLIGHT="20261126T211301_3a7cd8be"
PREFLIGHT_SHA="21b112c5faaee0c410a5045ab295b57a41c875eb6dda78f268c8b4c0cfb0d99b"
AUGMENTATION="20261126T180155_005d9a59"
AUGMENTATION_SHA="d5e7f6cb8f6afce7913a8a1cb9275889525137516e7386e85029674935a215fe"
BINS=[0,5,10,20,40,60,80,100,np.inf]
POLICIES=("A_historical","B_archived_smoter","C_tail_only","C_bounded_transfer","D_bounded_plus_smoter")
NEW=POLICIES[2:]


def transfer_weights(historical_y,candidate_y,tail_only=False):
    """Fixed experimental weights; preserve historic band mass; never force uniformity."""
    h=pd.cut(historical_y,BINS)
    c=pd.cut(candidate_y,BINS)
    if (h.isna().any() or c.isna().any() or not np.isfinite(historical_y).all()
            or not np.isfinite(candidate_y).all()):
        raise ValueError("Nonpositive or invalid target")
    out=pd.Series(0.,index=candidate_y.index)
    for band in c.cat.categories:
        mask=c.eq(band)
        n=int(mask.sum())
        if not n:
            continue
        tail=band.left>=40
        if tail_only and not tail:
            continue
        cap=.25 if tail else .10
        # Total candidate weight per band <=10% of historic TRAIN weight.
        out.loc[mask]=min(cap,.10*int(h.eq(band).sum())/n)
    return out


def load_candidates(digest):
    root=RESULTS_DIR/"generalization/model_1_directed_transfer_preflight"/PREFLIGHT
    m,_=verified_run(root,"directed_transfer_preflight",PREFLIGHT_SHA)
    chen=RESULTS_DIR/"generalization/model_1_chen_overlap_audit"/CHEN_RUN
    cm,_=verified_run(chen,"chen_overlap_screen",CHEN_MANIFEST)
    if m["source_sha256"]!=digest or cm["source_sha256"]!=digest or m["chen_manifest_sha256"]!=CHEN_MANIFEST:
        raise ValueError("Source identity differs")
    if file_sha256(ROOT/SOURCE/NAME)!=SHA or cm["original_sha256"]!=SHA:
        raise ValueError("Original differs")
    rows=pd.read_csv(chen/"row_review.csv")
    rows.index=pd.Index("chen:"+rows.excel_row.astype(str),name="candidate_id")
    flags=pd.read_csv(root/"fold_review.csv")
    flags=flags.loc[flags.source_dataset.eq(SOURCE)].copy()
    if (len(rows)!=3680 or not rows.index.is_unique or len(flags)!=5*len(rows)
            or flags.duplicated(["fold","candidate_id"]).any() or set(flags.fold)!=set(range(1,6))
            or set(flags.candidate_id)!=set(rows.index)):
        raise ValueError("Candidate identities differ")
    for fold,part in flags.groupby("fold"):
        part=part.set_index("candidate_id").reindex(rows.index)
        if not np.allclose(part.Li_icpms,rows.Li_icpms,rtol=0,atol=1e-8):
            raise ValueError("Candidate targets differ")
        allowed=part.fold_transfer_review_candidate.eq(True)
        expected=[1505,1560,1529,1559,1539][fold-1]
        if int(allowed.sum())!=expected or not rows.loc[allowed,"transfer_review_candidate"].eq(True).all():
            raise ValueError("Preflight membership differs")
        if part.loc[allowed,["test_group_overlap","test_publication_alias_overlap"]].any().any():
            raise ValueError("Candidate overlaps TEST")
    return rows,flags


def candidate_features(rows,x,measured):
    # Unknown geology/category/analytical methods stay missing; no geographic
    # extrapolation of categories. Canonical concentration aliases are research-only.
    cx=rows.reindex(columns=x.columns).copy()
    cx=add_geochemical_features(cx,version=2).reindex(columns=x.columns)
    rec=[v for v in measured if v.startswith("georoc_")]
    other=[v for v in measured if v not in rec]
    extra=pd.DataFrame(np.nan,index=cx.index,columns=other)
    mm,rr=measurement_blocks(pd.concat([cx,extra],axis=1),rows.reindex(columns=rec),cx)
    return pd.concat([cx,mm,rr],axis=1)


def archived_smoter(x,pilot,digest):
    root=RESULTS_DIR/"generalization/model_1_enriched_augmentation"/AUGMENTATION
    if file_sha256(root/"manifest.json")!=AUGMENTATION_SHA:
        raise ValueError("Synthetic experiment manifest differs")
    m=read(root/"manifest.json")
    if m["state"]!="complete" or m["source_sha256"]!=digest or m["feature_manifest_sha256"]!=FEATURE_SHA:
        raise ValueError("Synthetic experiment identity differs")
    for name,sha in m["outputs"].items():
        if file_sha256(root/name)!=sha:
            raise ValueError("Synthetic evidence differs: "+name)
    p=pd.read_csv(root/"predictions.csv",dtype={"row_id":str})
    p=p.loc[p.policy.eq("smoter_enriched")].sort_values("row_id").reset_index(drop=True)
    if (not p.row_id.is_unique or not p[["row_id","fold","group"]].equals(pilot[["row_id","fold","group"]])
            or not np.allclose(p.truth,pilot.truth,rtol=0,atol=1e-8)
            or not np.isfinite(p.prediction).all() or p.prediction.lt(0).any()):
        raise ValueError("Archived synthetic TEST differs")
    return p.set_index("row_id").reindex(x.index.astype(str))


def run(allow_unverified_metadata=False):
    if not allow_unverified_metadata:
        raise ValueError("Explicit --allow-unverified-metadata required; development sensitivity only")
    source=local_training_source("model_1")
    digest=file_sha256(source)
    prepared,keep,domain=cohort(source)
    x,y,groups,frame=(v.loc[keep] for v in (prepared.x,prepared.y,prepared.groups,prepared.frame))
    if len(x)!=3263 or len(x.columns)!=33 or len(prepared.x)!=3420:
        raise ValueError("Frozen cohort differs")
    splits=list(KFold(5,shuffle=True,random_state=SEED).split(x))
    pilot,_=check_pilot(RESULTS_DIR/"generalization/model_1_oasis_like"/PILOT,x,y,groups,splits,digest,domain)
    a,archive_features=load_reference(x,pilot,digest)
    b=archived_smoter(x,pilot,digest)
    r,m=load_recovery(prepared,digest)
    if not r.academic_member.eq(keep).all():
        raise ValueError("Recovery membership differs")
    measured,ratios=measurement_blocks(frame,r.loc[keep,m["measurement_columns"]],x)
    features=pd.concat([x,measured,ratios],axis=1)
    sources=source_series(frame)
    numeric,categorical=list(prepared.schema.numeric),list(prepared.schema.categorical)
    validate_training_schema(prepared.target,numeric+list(measured)+list(ratios),categorical)
    rows,flags=load_candidates(digest)
    cx=candidate_features(rows,x,measured).reindex(columns=features.columns)
    for name in categorical:
        cx[name]=cx[name].astype(object).where(cx[name].notna(),np.nan)
    cy=rows.Li_icpms.astype(float)
    out=RESULTS_DIR/"generalization/model_1_data_closure"/(
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")+"_"+uuid4().hex[:8])
    out.mkdir(parents=True,exist_ok=False)
    manifest={"state":"running","scope":"fixed_academic_development_data_closure",
        "source_sha256":digest,"domain":domain,"feature_manifest_sha256":FEATURE_SHA,
        "recovery_manifest_sha256":RECOVERY_SHA,"preflight_manifest_sha256":PREFLIGHT_SHA,
        "synthetic_manifest_sha256":AUGMENTATION_SHA,"policies":POLICIES,"max_new_fits":15,
        "metadata_certified":False,"direct_comparability":False,"independent_holdout":False,
        "reused_development_folds":True,"retrospective_target_assisted_recovery":True,
        "no_inner_selection":True,"new_predictors":False,"approved_new_rows":0,
        "models_saved":False,"promotion":False,"release_eligible":False}
    write_legacy_evidence_json(out/"manifest.json",manifest)
    device=detect_device(prefer_gpu=False)
    params={k:v for k,v in PROFILES["hist_gradient_boosting"].items() if k!="treatment_policy"}
    params["min_samples_leaf"]=5
    predictions,metrics,members,lineage,coverage,partitions,synthetic=[],[],[],[],[],[],[]
    fits=0
    try:
        for fold,(tr,te) in enumerate(splits,1):
            extra=select_columns(measured.iloc[tr],groups.iloc[tr])+select_columns(ratios.iloc[tr],groups.iloc[tr])
            schema=FeatureSchema(numeric+extra,categorical,2,"stable_v1")
            cols=schema.numeric+schema.categorical
            if cols!=archive_features[fold] or not a.iloc[te].fold.eq(fold).all() or not b.iloc[te].fold.eq(fold).all():
                raise ValueError("Archived control feature or TEST identity differs")
            tx,ty=features.iloc[tr],y.iloc[tr]
            eligible=flags.loc[flags.fold.eq(fold)&flags.fold_transfer_review_candidate.eq(True),"candidate_id"]
            if set(rows.loc[eligible,"spatial_group"])&set(groups.iloc[te]):
                raise ValueError("New candidates overlap historical TEST groups")
            if not cy.loc[eligible].between(domain["q025"],domain["q975"]).all():
                raise ValueError("Candidate outside frozen academic target range")
            sx,sy,_,parents,_=augment(tx,ty,groups.iloc[tr],sources.iloc[tr],
                list(x.columns),list(measured),categorical,"smoter_enriched",SEED+100*fold)
            allowed=set(tx.index.astype(str))
            if any(str(v[k]) not in allowed for v in parents for k in ("parent_a","parent_b") if k in v):
                raise ValueError("Synthetic parent outside TRAIN")
            lineage.extend({"fold":fold,"synthetic_position":i,**v} for i,v in enumerate(parents))
            generated=sx.iloc[len(ty):].copy().reset_index(drop=True)
            generated.insert(0,"synthetic_position",np.arange(len(generated)))
            generated.insert(0,"fold",fold)
            generated["target_li_ppm"]=sy.iloc[len(ty):].to_numpy(float)
            synthetic.append(generated)
            partitions.append({"fold":fold,"train_ids":tx.index.astype(str).tolist(),
                "test_ids":x.index[te].astype(str).tolist(),"features":cols})
            values={POLICIES[0]:a.iloc[te].prediction.to_numpy(float),
                    POLICIES[1]:b.iloc[te].prediction.to_numpy(float)}
            for policy in NEW:
                w=transfer_weights(ty,cy.loc[eligible],tail_only=policy=="C_tail_only")
                w=w.loc[w.gt(0)]
                base_x,base_y=(sx,sy) if policy==NEW[-1] else (tx,ty)
                fit_x=pd.concat([base_x[cols],cx.loc[w.index,cols]],ignore_index=True)
                fit_y=pd.concat([base_y,cy.loc[w.index]],ignore_index=True)
                weights=np.concatenate([np.ones(len(base_y)),w.to_numpy(float)])
                if len(fit_x)!=len(fit_y) or not np.isfinite(weights).all() or (weights<=0).any():
                    raise ValueError("Fit membership or weights invalid")
                model=build_pipeline("hist_gradient_boosting",device,SEED,schema,params)
                model.set_params(preprocess__numeric="passthrough")
                with estimator_thread_scope():
                    model.fit(fit_x,fit_y,model__sample_weight=weights)
                fits+=1
                values[policy]=np.maximum(np.asarray(model.predict(features.iloc[te][cols]),float),0.)
                members.extend({"fold":fold,"policy":policy,"candidate_id":i,"Li_icpms":float(cy.loc[i]),
                    "weight":float(v),"spatial_group":str(rows.loc[i,"spatial_group"]),"publication":str(rows.loc[i,"source_publication"])}
                    for i,v in w.items())
                bands=pd.cut(fit_y,BINS)
                for band in bands.cat.categories:
                    hit=bands.eq(band).to_numpy()
                    coverage.append({"fold":fold,"policy":policy,"band":str(band),
                        "historical_rows":int(pd.cut(ty,BINS).eq(band).sum()),
                        "synthetic_rows":int(pd.cut(base_y.iloc[len(ty):],BINS).eq(band).sum()),
                        "candidate_rows":int(pd.cut(cy.loc[w.index],BINS).eq(band).sum()),
                        "fit_rows":int(hit.sum()),"effective_weight":float(weights[hit].sum())})
            for policy,pred in values.items():
                if pred.shape!=(len(te),) or not np.isfinite(pred).all() or (pred<0).any():
                    raise ValueError("Invalid evaluation predictions")
                mm=regression_metrics(y.iloc[te],pred).to_dict()
                metrics.append({"fold":fold,"policy":policy,**mm})
                predictions.extend({"fold":fold,"policy":policy,"row_id":str(x.index[i]),
                    "group":str(groups.iloc[i]),"truth":float(y.iloc[i]),"prediction":float(v)}
                    for i,v in zip(te,pred,strict=True))
                print(f"Fold {fold}/5 | {policy} | RMSE={mm['rmse']:.4f} | R2={mm['r2']:.4f}",flush=True)
            write_table(out/"predictions.csv",pd.DataFrame(predictions))
            write_table(out/"fold_metrics.csv",pd.DataFrame(metrics))
            write_table(out/"candidate_membership.csv",pd.DataFrame(members))
            write_table(out/"training_band_coverage.csv",pd.DataFrame(coverage))
            write_table(out/"synthetic_features.csv",pd.concat(synthetic,ignore_index=True))
            write_legacy_evidence_json(out/"synthetic_lineage.json",lineage)
            write_legacy_evidence_json(out/"partitions.json",partitions)
        report,band_metrics=[],[]
        for policy,p in pd.DataFrame(predictions).groupby("policy"):
            p=p.sort_values("row_id").reset_index(drop=True)
            if (not p.row_id.is_unique or not p[["row_id","fold","group"]].equals(pilot[["row_id","fold","group"]])
                    or not np.allclose(p.truth,pilot.truth,atol=1e-8,rtol=0)):
                raise ValueError("OOF identities differ")
            fm=pd.DataFrame([regression_metrics(g.truth,g.prediction).to_dict() for _,g in p.groupby("fold")])
            rmse,r2,mae=map(float,(fm.rmse.mean(),fm.r2.mean(),fm.mae.mean()))
            tail=p.loc[p.truth.gt(40)]
            report.append({"policy":policy,"rmse_mean":rmse,"r2_mean":r2,"mae_mean":mae,
                "rmse_global":regression_metrics(p.truth,p.prediction).rmse,
                "tail_rmse":regression_metrics(tail.truth,tail.prediction).rmse,
                "tail_bias":float((tail.prediction-tail.truth).mean()),**benchmark_status(rmse,r2)})
            for band,g in p.groupby(pd.cut(p.truth,BINS),observed=True):
                band_metrics.append({"policy":policy,"band":str(band),"rows":len(g),
                    **regression_metrics(g.truth,g.prediction).to_dict()})
        if fits!=15 or len(report)!=5 or file_sha256(source)!=digest:
            raise ValueError("Closure campaign incomplete")
        base=next(v for v in report if v["policy"]==POLICIES[0])
        for v in report:
            v["improves_control_rmse"]=v["rmse_mean"]<base["rmse_mean"]-1e-10
            v["tail_not_worse"]=v["tail_rmse"]<=base["tail_rmse"]+1e-10
            v["mae_not_materially_worse"]=v["mae_mean"]<=base["mae_mean"]+.1
            v["numerical_closure_target_met"]=bool(v["both_better"] and v["improves_control_rmse"]
                and v["tail_not_worse"] and v["mae_not_materially_worse"])
        write_table(out/"summary.csv",pd.DataFrame(report))
        write_table(out/"band_metrics.csv",pd.DataFrame(band_metrics))
        manifest["summary"]=report
        manifest["state"]="complete"
        manifest["numerical_target_policies"]=[v["policy"] for v in report if v["numerical_closure_target_met"]]
        manifest["decision"]="development_comparison_only_no_automatic_selection_or_release"
    except BaseException as exc:
        manifest["state"]="interrupted"
        manifest["error"]=str(exc)
        raise
    finally:
        manifest["new_fits"]=fits
        manifest["outputs"]={p.name:file_sha256(p) for p in out.iterdir() if p.is_file() and p.name!="manifest.json"}
        write_legacy_evidence_json(out/"manifest.json",manifest)
    print(pd.DataFrame(report).to_string(index=False))
    print(f"Estado: {manifest['state']} | Evidencias: {out}",flush=True)
    return out


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-unverified-metadata",action="store_true")
    run(parser.parse_args().allow_unverified_metadata)
