"""Read-only inventory of additional geochemistry for M1.

No fitting, feature selection, acquisition, or change to the evaluation cohort.
Unknown chemical aliases require explicit human review; never use identifiers,
analytical lithium aliases, targets, model outputs or geochemical totals.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from uuid import uuid4

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.paths import RESULTS_DIR
from lithiumscope.core.scientific.eda_statistics import source_series
from lithiumscope.datasets.management.sources.local import local_training_source
from lithiumscope.experiments.model_1.protocols.cohorts import cohort
from lithiumscope.experiments.model_1.studies.expansion.academic_bridge import PILOT, SEED, check_pilot
from lithiumscope.experiments.model_1.data_expansion.sensitivity_data import read
from lithiumscope.model_1.preparation.detection_limits import clean_detection_limits
from lithiumscope.model_1.schema import validate_training_schema
from lithiumscope.results.atomic_exports import write_table, write_legacy_evidence_json

REFINEMENT="20261126T161005_a8008a65"
REFINEMENT_SHA="da784283574b77281c0bee1ef85b80b9a4da0843ab7d26b584497092ac814a18"
from lithiumscope.experiments.data_lab.source_studies.geochemistry import ELEMENTS
AGE={"Age (Ma)","Age (ma)","Age","age_ma"}
# These ratios use only known independently measured predictors.
RATIOS=(
    ("Rb_icpms","Cs_icpms"),("Rb_icpms","Sr_icpms"),
    ("Rb_icpms","Ba_icpms"),("Th_icpms","U_icpms"),
    ("Nb_icpms","Ta_icpms"),("Zr_icpms","Hf_icpms"),
    ("K2O","Na2O"),("Cs_icpms","Ba_icpms"),("Cs_icpms","Sr_icpms"),
    ("Th_icpms","Nb_icpms"),("Sr_icpms","Y_icpms"),
    ("La_icpms","Yb_icpms"),("La_icpms","Sm_icpms"),
    ("Gd_icpms","Yb_icpms"),("Nb_icpms","U_icpms"),
)


def classify(name, existing):
    if name in existing:
        return "existing"
    if name in AGE:
        return "age_review"
    normalized=str(name).strip().casefold()
    match=re.fullmatch(r"([a-z]{1,2})[\s_]*(icpms|xrf|ppm|ppb|\(ppm\)|\(ppb\))",normalized)
    if match and match[1] in ELEMENTS:
        return "explicit_element_measurement_review"
    if normalized in ELEMENTS:
        return "element_unit_unknown_review"
    return "not_allowlisted"


def numeric_measurement(values):
    converted=pd.to_numeric(values,errors="coerce")
    return converted.where(np.isfinite(converted)&converted.ge(0)).astype(float)


def derived(frame):
    result={}
    for a,b in RATIOS:
        if a in frame and b in frame:
            left,right=numeric_measurement(frame[a]),numeric_measurement(frame[b])
            result["log_ratio_"+a+"_"+b]=np.log(left.where(left.gt(0)))-np.log(right.where(right.gt(0)))
    # Oxides in wt%: molar A/CNK, separate from canonical mass proxy.
    names=["Al2O3","CaO","Na2O","K2O"]
    if set(names).issubset(frame):
        a,c,n,k=[numeric_measurement(frame[name]) for name in names]
        den=c/56.077+n/61.979+k/94.196
        result["molar_A_CNK"]=(a/101.961)/den.where(den.gt(0))
    return pd.DataFrame(result,index=frame.index).replace([np.inf,-np.inf],np.nan)


def profile(values,y,groups):
    valid=values.notna()
    vv=values[valid]
    tt=y[valid]
    correlation=vv.corr(tt,method="spearman") if len(vv)>=10 and vv.nunique()>1 and tt.nunique()>1 else np.nan
    return {"rows":len(values),"observed":int(valid.sum()),"coverage":float(valid.mean()),
        "groups_observed":int(groups[valid].nunique()),"unique":int(vv.nunique()),
        "zero":int(vv.eq(0).sum()),"q05":vv.quantile(.05),"median":vv.median(),"q95":vv.quantile(.95),
        "spearman_descriptive":correlation}


def run():
    source=local_training_source("model_1")
    digest=file_sha256(source)
    prepared,keep,domain=cohort(source)
    if len(prepared.x)!=3420 or int(keep.sum())!=3263 or len(prepared.x.columns)!=33:
        raise ValueError("Frozen cohort differs")
    x,y,groups,frame=(v.loc[keep].copy() for v in (prepared.x,prepared.y,prepared.groups,prepared.frame))
    splits=list(KFold(5,shuffle=True,random_state=SEED).split(x))
    pilot,_=check_pilot(RESULTS_DIR/"generalization/model_1_oasis_like"/PILOT,x,y,groups,splits,digest,domain)
    path=RESULTS_DIR/"generalization/model_1_expansion_academic_refinement"/REFINEMENT
    manifest_path=path/"manifest.json"
    if file_sha256(manifest_path)!=REFINEMENT_SHA:
        raise ValueError("Refinement manifest differs")
    m=read(manifest_path)
    if m["source_sha256"]!=digest or m["state"]!="complete":
        raise ValueError("Refinement evidence differs")
    if file_sha256(path/"predictions.csv")!=m["outputs"]["predictions.csv"]:
        raise ValueError("Refinement predictions differ")
    pred=pd.read_csv(path/"predictions.csv",dtype={"row_id":str}).sort_values("row_id").reset_index(drop=True)
    if (not pred.row_id.is_unique or not pred[["row_id","fold","group"]].equals(pilot[["row_id","fold","group"]])
            or not np.allclose(pred.truth,pilot.truth,rtol=0,atol=1e-8)):
        raise ValueError("Academic identities differ")
    predictions=pred.set_index("row_id").prediction.reindex(x.index.astype(str))
    if not np.isfinite(predictions).all():
        raise ValueError("Incomplete prediction evidence")
    residual=pd.Series(y.to_numpy()-predictions.to_numpy(),index=x.index)
    inventory=[]
    additional={}
    for name in frame.columns:
        role=classify(name,x.columns)
        # Report only counts for unapproved columns; do not export their values.
        row={"column":name,"role":role,"dtype":str(frame[name].dtype),
             "present":int(frame[name].notna().sum()),"rows":len(frame)}
        if role in {"explicit_element_measurement_review","age_review","element_unit_unknown_review"}:
            values=frame[name]
            if role=="explicit_element_measurement_review":
                values=clean_detection_limits(frame[[name]],columns=[name],version=2)[name]
            values=numeric_measurement(values)
            row.update(profile(values,y,groups))
            # Unknown unit or age alias is inventory-only, not an extra feature.
            if role=="explicit_element_measurement_review" and values.notna().any():
                additional[name]=values
        inventory.append(row)
    extra=pd.DataFrame(additional,index=x.index)
    validate_training_schema(prepared.target,list(extra.columns),[])
    measured=pd.concat([x,extra],axis=1)
    ratios=derived(measured)
    candidate=pd.concat([extra,ratios],axis=1)
    validate_training_schema(prepared.target,list(candidate.columns),[])
    source_labels=source_series(frame)
    bands=pd.cut(y,[0,5,10,20,40,60,80,100,np.inf])
    profiles,by_source,by_band,fold_train=[],[],[],[]
    for name in candidate:
        values=candidate[name]
        p=profile(values,y,groups)
        valid=values.notna()
        p.update(feature=name,kind="measured_review" if name in extra else "derived_review",
            residual_spearman_descriptive=values[valid].corr(residual[valid],method="spearman")
                if valid.sum()>=10 and values[valid].nunique()>1 and residual[valid].nunique()>1 else np.nan)
        profiles.append(p)
        for label,idx in source_labels.groupby(source_labels).groups.items():
            by_source.append({"feature":name,"source":str(label),**profile(values.loc[idx],y.loc[idx],groups.loc[idx])})
        for label,idx in bands.groupby(bands,observed=True).groups.items():
            by_band.append({"feature":name,"band":str(label),**profile(values.loc[idx],y.loc[idx],groups.loc[idx])})
        for fold,(tr,_) in enumerate(splits,1):
            fold_train.append({"feature":name,"fold":fold,**profile(values.iloc[tr],y.iloc[tr],groups.iloc[tr])})
    out=RESULTS_DIR/"generalization/model_1_expansion_feature_audit"/(
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")+"_"+uuid4().hex[:8])
    out.mkdir(parents=True,exist_ok=False)
    outputs={"column_inventory.csv":pd.DataFrame(inventory),"feature_profiles.csv":pd.DataFrame(profiles),
        "coverage_by_source.csv":pd.DataFrame(by_source),"coverage_by_band.csv":pd.DataFrame(by_band),
        "outer_train_profiles.csv":pd.DataFrame(fold_train)}
    for name,table in outputs.items():
        write_table(out/name,table)
    if file_sha256(source)!=digest:
        raise ValueError("Source changed")
    write_legacy_evidence_json(out/"manifest.json",{"state":"review_required","scope":"m1_feature_inventory_only",
        "source_sha256":digest,"refinement_manifest_sha256":REFINEMENT_SHA,"rows":len(x),
        "canonical_features":list(x.columns),"additional_measurement_columns":list(extra.columns),
        "derived_columns":list(ratios.columns),"domain":domain,"training":False,"feature_selection":False,
        "production_schema_changed":False,"acquisition":False,"promotion":False,
        "full_cohort_profiles_descriptive_only":True,"outputs":{name:file_sha256(out/name) for name in outputs}})
    print(pd.DataFrame(profiles).to_string(index=False))
    print(f"Estado: review_required | Evidencias: {out}",flush=True)
    return out


if __name__=="__main__":
    run()
