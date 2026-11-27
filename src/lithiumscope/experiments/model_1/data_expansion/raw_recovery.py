"""Recover omitted GEOROC measurements without rebuilding the frozen dataset.

Only unique, chemically corroborated matches to original historical partitions
are emitted. Explicit ppm/ppb fields remain separate from Mamani ICP-MS/XRF.
No training, download, target change or production schema mutation.
"""
from __future__ import annotations

import re
from pathlib import Path
from datetime import datetime, timezone
from uuid import uuid4

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from lithiumscope.core.config import load_config
from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.paths import PROJECT_ROOT, RESULTS_DIR, DATA_DIR
from lithiumscope.core.scientific.eda_statistics import source_series
from lithiumscope.datasets.adapters.georoc.schema import infer_georoc_column_map, MAJOR_OXIDES
from lithiumscope.datasets.adapters.georoc.transformation import harmonize_chunk
from lithiumscope.datasets.readers.georoc_raw_reader import detect_georoc_encoding
from lithiumscope.experiments.data_lab.source_studies.geochemistry import measurement_map, numeric_ppm
from lithiumscope.datasets.management.sources.local import local_training_source
from lithiumscope.experiments.model_1.protocols.cohorts import cohort
from lithiumscope.experiments.model_1.studies.expansion.academic_bridge import PILOT, SEED, check_pilot
from lithiumscope.experiments.model_1.data_expansion.feature_audit import profile
from lithiumscope.experiments.model_1.data_expansion.sensitivity_data import read
from lithiumscope.results.atomic_exports import write_table, write_legacy_evidence_json

AUDIT="20261126T162414_9f0900a7"
AUDIT_SHA="3c4ff8ddb1e9dc5004c951ae9cdeb1e10cad1bed11c311c47166831f04bc91d6"
EXPANSION="20261126T145129_ee7f5431"
EXPANSION_SHA="0817aac5e5d869e9be0ec9356c8eea189278cf8ec1f896d19e025384affac6b2"
IDENTITY=["Latitude","Longitude","Li_icpms"]
CHEM=[c for c in MAJOR_OXIDES if c!="Fe2O3"]+[
    "Th_icpms","U_icpms","Rb_icpms","Cs_icpms","Nb_icpms","Ta_icpms",
    "Pb_icpms","Ba_icpms","Sr_icpms","Zr_icpms","V_icpms","Hf_icpms"]
def normalized_sample(values):
    return values.astype("string").fillna("").str.strip().str.casefold()

def unique_matches(base,raw):
    """Require sample, coordinates, Li, >=5 agreeing independent predictors.
    Reject ambiguous matches and raw rows assigned to multiple historical rows.
    """
    matches=[]
    if not set(IDENTITY+["source_sample"]).issubset(raw.columns):
        return pd.DataFrame([{"row_id":str(i),"raw_position":None,"state":"raw_core_unavailable","shared_chemistry":0} for i in base.index])
    sample=normalized_sample(raw.source_sample)
    by_sample={k:v for k,v in sample.groupby(sample).groups.items() if k}
    numeric=raw.reindex(columns=IDENTITY+CHEM).apply(pd.to_numeric,errors="coerce")
    historic=base.reindex(columns=IDENTITY+CHEM).apply(pd.to_numeric,errors="coerce")
    for idx,name in normalized_sample(base.source_sample).items():
        candidates=by_sample.get(name,[])
        if not len(candidates):
            matches.append({"row_id":str(idx),"raw_position":None,"state":"sample_unmatched","shared_chemistry":0})
            continue
        original=historic.loc[idx]
        search=numeric.loc[candidates]
        core=np.isclose(search[IDENTITY].to_numpy(float),original[IDENTITY].to_numpy(float),
                        rtol=1e-8,atol=1e-7).all(axis=1)
        left=search[CHEM].to_numpy(float)
        right=original[CHEM].to_numpy(float)
        shared=np.isfinite(left)&np.isfinite(right)
        counts=shared.sum(axis=1)
        close=np.isclose(left,right,rtol=1e-6,atol=1e-6)
        eligible=core&(counts>=5)&(close|~shared).all(axis=1)
        positions=np.flatnonzero(eligible)
        if len(positions)==1:
            position=positions[0]
            matches.append({"row_id":str(idx),"raw_position":int(search.index[position]),
                            "state":"matched","shared_chemistry":int(counts[position])})
        else:
            matches.append({"row_id":str(idx),"raw_position":None,
                            "state":"ambiguous" if len(positions)>1 else "core_or_chemistry_mismatch",
                            "shared_chemistry":0})
    result=pd.DataFrame(matches)
    if not result.empty:
        repeated=result.state.eq("matched")&result.raw_position.duplicated(keep=False)
        result.loc[repeated,"state"]="raw_row_reused"
        result.loc[repeated,"raw_position"]=np.nan
    return result

def run():
    source=local_training_source("model_1")
    digest=file_sha256(source)
    prepared,keep,domain=cohort(source)
    if len(prepared.x)!=3420 or int(keep.sum())!=3263:
        raise ValueError("Frozen historical cohort differs")
    x,y,groups=(v.loc[keep] for v in (prepared.x,prepared.y,prepared.groups))
    splits=list(KFold(5,shuffle=True,random_state=SEED).split(x))
    check_pilot(RESULTS_DIR/"generalization/model_1_oasis_like"/PILOT,x,y,groups,splits,digest,domain)
    audit_path=RESULTS_DIR/"generalization/model_1_expansion_feature_audit"/AUDIT/"manifest.json"
    if file_sha256(audit_path)!=AUDIT_SHA:
        raise ValueError("Feature audit manifest differs")
    cfg=load_config("model_1")["data_sources"]["georoc"]
    pattern=PROJECT_ROOT/cfg["input_glob"]
    # The original ingestion audit is a separate file, not the combined source.
    ingest_path=(PROJECT_ROOT/cfg["harmonized_path"]).with_suffix(".audit.json")
    ingest=read(ingest_path) if ingest_path.is_file() else {}
    hashes={}
    for name,value in ingest.get("source_sha256",{}).items():
        basename=Path(str(name).replace("\\","/")).name
        hashes.setdefault(basename,set()).add(value)
    base=prepared.frame.copy()
    labels=source_series(base)
    external=base.loc[labels.eq("GEOROC")]
    if external.empty or not external.index.is_unique:
        raise ValueError("Historical GEOROC rows unavailable")
    recovered=pd.DataFrame(index=base.index.astype(str))
    recovered.index.name="row_id"
    inventory,match_tables=[],[]
    for name,rows in external.groupby("source_file",dropna=False):
        if pd.isna(name) or Path(str(name)).name!=name:
            inventory.append({"file":str(name),"state":"invalid_source_file","historical_rows":len(rows)})
            continue
        path=pattern.parent/name
        if not path.is_file():
            inventory.append({"file":name,"state":"missing","historical_rows":len(rows)})
            continue
        sha=file_sha256(path)
        expected=hashes.get(name,set())
        verified=expected=={sha}
        encoding=detect_georoc_encoding(path)
        header=list(pd.read_csv(path,nrows=0,encoding=encoding).columns)
        extras=measurement_map(header)
        mapping=infer_georoc_column_map(header)
        if cfg["iron_total_mode"]=="feot_equivalent":
            for column in header:
                if column.upper().replace(" ","").replace("_","") in {"FEOT(WT%)","FEOT"}:
                    mapping[column]="FeOT_total_as_FeO"
                    break
        inv={"file":name,"state":"verified" if verified else "raw_hash_unverified",
             "sha256":sha,"historical_rows":len(rows),"extra_headers":extras,
             "header":header,"matched":0}
        inventory.append(inv)
        if not verified:
            continue
        raw=pd.read_csv(path,usecols=list(dict.fromkeys([*mapping,*[v[0] for v in extras.values()]])),
                        encoding=encoding,dtype=str,low_memory=False)
        harmonized,_=harmonize_chunk(raw[list(mapping)],column_map=mapping,source_name="GEOROC",
            source_file=name,allowed_material_types=tuple(cfg["allowed_material_types"]),
            minimum_predictors=int(cfg["minimum_predictors"]),iron_total_mode=cfg["iron_total_mode"])
        matches=unique_matches(rows,harmonized)
        matches["file"]=name
        matches["raw_sha256"]=sha
        match_tables.append(matches)
        good=matches.loc[matches.state.eq("matched")]
        inv["matched"]=len(good)
        for element,(column,unit) in extras.items():
            field="georoc_"+element+"_ppm"
            if field not in recovered:
                recovered[field]=np.nan
            values=numeric_ppm(raw[column].loc[good.raw_position.astype(int)],unit)
            recovered.loc[good.row_id,field]=values.to_numpy()
        if file_sha256(path)!=sha:
            raise ValueError("Raw partition changed during recovery")
    # Historical filtered query may omit extra elements at acquisition. Reuse
    # verified Andean precompiled partitions, but never equate rounded coords
    # and Li alone with sample identity. Require the same exact match contract.
    research=RESULTS_DIR/"model_1/expansion"/EXPANSION
    mp=research/"manifest.json"
    research_matches=[]
    research_values=[]
    if mp.is_file():
        if file_sha256(mp)!=EXPANSION_SHA:
            raise ValueError("Expansion manifest differs")
        em=read(mp)
        ip=research/"raw_metadata_inventory.json"
        if em.get("source_sha256")!=digest or file_sha256(ip)!=em["outputs"][ip.name]:
            raise ValueError("Research inventory differs")
        for item in read(ip):
            if item["source"]!="georoc_andean_202609":
                continue
            name=item["file"]
            if Path(name).name!=name:
                raise ValueError("Invalid research partition name")
            path=DATA_DIR/"raw/model_1/research"/item["source"]/name
            inv={"file":name,"state":"research_missing","historical_rows":len(external),"matched":0}
            inventory.append(inv)
            if not path.is_file():
                continue
            sha=file_sha256(path)
            if sha!=item["sha256"] or item["raw_state"]!="verified":
                inv.update(state="research_hash_unverified",sha256=sha)
                continue
            encoding=detect_georoc_encoding(path)
            header=list(pd.read_csv(path,nrows=0,encoding=encoding).columns)
            extras=measurement_map(header)
            mapping=infer_georoc_column_map(header)
            inv.update(state="research_verified",sha256=sha,extra_headers=extras,header=header)
            raw=pd.read_csv(path,usecols=list(dict.fromkeys([*mapping,*[v[0] for v in extras.values()]])),
                            encoding=encoding,dtype=str,low_memory=False)
            harmonized,_=harmonize_chunk(raw[list(mapping)],column_map=mapping,source_name=item["source"],
                source_file=name,allowed_material_types=tuple(cfg["allowed_material_types"]),
                minimum_predictors=int(cfg["minimum_predictors"]),iron_total_mode="original")
            matches=unique_matches(external,harmonized)
            matches["file"]=name
            matches["raw_sha256"]=sha
            matches["match_origin"]="research"
            research_matches.append(matches)
            good=matches.loc[matches.state.eq("matched")]
            inv["matched"]=len(good)
            part=pd.DataFrame({"row_id":good.row_id.to_numpy(),"file":name})
            for element,(column,unit) in extras.items():
                part["georoc_"+element+"_ppm"]=numeric_ppm(raw[column].loc[good.raw_position.astype(int)],unit).to_numpy()
            research_values.append(part)
            if file_sha256(path)!=sha:
                raise ValueError("Research partition changed during recovery")
    if research_matches:
        all_matches=pd.concat(research_matches,ignore_index=True)
        # A raw ambiguity in any partition also precludes assignment.
        good=all_matches.loc[all_matches.state.eq("matched")]
        blocked=set(all_matches.loc[all_matches.state.isin(["ambiguous","raw_row_reused"]),"row_id"])
        blocked.update(good.loc[good.row_id.duplicated(keep=False),"row_id"])
        bad=all_matches.state.eq("matched")&all_matches.row_id.isin(blocked)
        all_matches.loc[bad,"state"]="cross_partition_ambiguous"
        all_matches.loc[bad,"raw_position"]=np.nan
        match_tables.append(all_matches)
        joined=pd.concat(research_values,ignore_index=True)
        joined=joined.loc[~joined.row_id.isin(blocked)].set_index("row_id").drop(columns="file")
        if not joined.index.is_unique:
            raise ValueError("Research recovery ambiguous")
        # Preserve exact original values, and don't silently resolve disagreement.
        for field in joined:
            if field not in recovered:
                recovered[field]=np.nan
            old=recovered.loc[joined.index,field]
            new=joined[field]
            conflict=old.notna()&new.notna()&~np.isclose(old,new,rtol=1e-6,atol=1e-6)
            recovered.loc[joined.index,field]=old.where(old.notna(),new).where(~conflict)
    recovered.insert(0,"academic_member",recovered.index.isin(x.index.astype(str)))
    academic=recovered.loc[x.index.astype(str)].drop(columns="academic_member")
    academic.index=x.index
    records,bands,folds=[],[],[]
    for name in academic:
        values=academic[name]
        records.append({"feature":name,**profile(values,y,groups)})
        for band,idx in pd.cut(y,[0,5,10,20,40,60,80,100,np.inf]).groupby(
                pd.cut(y,[0,5,10,20,40,60,80,100,np.inf]),observed=True).groups.items():
            bands.append({"feature":name,"band":str(band),**profile(values.loc[idx],y.loc[idx],groups.loc[idx])})
        for fold,(tr,_) in enumerate(splits,1):
            folds.append({"feature":name,"fold":fold,**profile(values.iloc[tr],y.iloc[tr],groups.iloc[tr])})
    out=RESULTS_DIR/"generalization/model_1_expansion_raw_recovery"/(
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")+"_"+uuid4().hex[:8])
    out.mkdir(parents=True,exist_ok=False)
    tables={"recovered_measurements.csv":recovered.reset_index(),
            "row_matches.csv":pd.concat(match_tables,ignore_index=True) if match_tables else pd.DataFrame(
                columns=["row_id","raw_position","state","shared_chemistry","file","raw_sha256"]),
            "feature_profiles.csv":pd.DataFrame(records,columns=["feature","rows","observed","coverage",
                "groups_observed","unique","zero","q05","median","q95","spearman_descriptive"]),
            "coverage_by_band.csv":pd.DataFrame(bands),
            "outer_train_profiles.csv":pd.DataFrame(folds)}
    for name,table in tables.items():
        write_table(out/name,table)
    write_legacy_evidence_json(out/"raw_inventory.json",inventory)
    if file_sha256(source)!=digest:
        raise ValueError("Historical source changed")
    write_legacy_evidence_json(out/"manifest.json",{"state":"review_required","scope":"m1_raw_measurements_recovery",
        "source_sha256":digest,"feature_audit_sha256":AUDIT_SHA,"historical_rows":len(base),
        "academic_rows":len(x),"georoc_rows":len(external),"matched_rows":int(pd.concat(match_tables,ignore_index=True).loc[lambda t:t.state.eq("matched"),"row_id"].nunique()) if match_tables else 0,
        "research_expansion_sha256":EXPANSION_SHA if mp.is_file() else None,
        "ingestion_audit_sha256":file_sha256(ingest_path) if ingest_path.is_file() else None,
        "measurement_columns":list(academic.columns),"methods_certified":False,"training":False,
        "production_schema_changed":False,"acquisition":False,"promotion":False,
        "outputs":{name:file_sha256(out/name) for name in [*tables,"raw_inventory.json"]}})
    print(pd.DataFrame(inventory).drop(columns=["header","extra_headers"],errors="ignore").to_string(index=False))
    print(pd.DataFrame(records).to_string(index=False))
    print(f"Estado: review_required | Evidencias: {out}",flush=True)
    return out


if __name__=="__main__":
    run()
