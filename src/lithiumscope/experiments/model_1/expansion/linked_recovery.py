"""Recover additional chemistry through already audited one-to-one historical links."""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.paths import RESULTS_DIR, DATA_DIR
from lithiumscope.datasets.management.sources.local import local_training_source
from lithiumscope.datasets.adapters.georoc.schema import infer_georoc_column_map
from lithiumscope.datasets.adapters.georoc.transformation import harmonize_chunk
from lithiumscope.datasets.readers.georoc_raw_reader import detect_georoc_encoding
from lithiumscope.experiments.data_lab.candidate_screening import sample, num, CHEM
from lithiumscope.experiments.model_1.balanced.academic import cohort
from lithiumscope.experiments.model_1.expansion.academic_bridge import PILOT, SEED, check_pilot
from lithiumscope.experiments.model_1.expansion.raw_recovery import EXPANSION, EXPANSION_SHA, measurement_map, numeric_ppm
from lithiumscope.experiments.model_1.expansion.feature_audit import profile
from lithiumscope.experiments.model_1.expansion.sensitivity_data import load_evidence, read
from lithiumscope.results.atomic_exports import write_table, write_legacy_evidence_json

PROVENANCE_SHA="0bb72d0273c5c2d5c25c7b8b2fdeb406b6e08d4bd2bb724b0ae453aa762abd7b"
ANDEAN="georoc_andean_202609"
CORE=["Latitude","Longitude","Li_icpms"]
CHECK_CHEM=[c for c in CHEM if c!="Fe2O3"]


def one_to_one(matches):
    links=matches.loc[matches.candidate_source.eq(ANDEAN)&matches.base_prepared.eq(True)
        &matches.base_source.eq("georoc")&matches.same_sample_name.eq(True)
        &matches.chemistry_agreement.eq(True)&matches.shared_chemistry.ge(5)].copy()
    # Reject all ambiguous edges before selecting an individual raw partition.
    unique=~links.base_row_position.duplicated(keep=False)&~links.candidate_id.duplicated(keep=False)
    return links.loc[unique].copy(),links.loc[~unique].copy()


def corroborate(raw, candidate, historic):
    """Aligned frames, independently recheck raw->candidate and raw->historical."""
    if not raw.index.equals(candidate.index) or not raw.index.equals(historic.index):
        raise ValueError("Unaligned raw identities")
    s=sample(raw)
    names=s.ne("")&s.eq(sample(candidate))&s.eq(sample(historic))
    a=num(raw,CORE).to_numpy(float)
    def key_match(other):
        b=num(other,CORE).to_numpy(float)
        return (np.isfinite(a).all(axis=1)&np.isfinite(b).all(axis=1)
            &(np.round(a[:,:2],5)==np.round(b[:,:2],5)).all(axis=1)
            &(np.round(a[:,2],6)==np.round(b[:,2],6)))
    ac=num(raw,CHECK_CHEM).to_numpy(float)
    def chemistry(other,strict):
        bc=num(other,CHECK_CHEM).to_numpy(float)
        shared=np.isfinite(ac)&np.isfinite(bc)
        close=(np.isclose(ac,bc,rtol=1e-6,atol=1e-6) if strict
               else np.abs(ac-bc)<=np.maximum(.001,.02*np.maximum(np.abs(ac),np.abs(bc))))
        return shared.sum(axis=1),(shared.sum(axis=1)>=5)&(close|~shared).all(axis=1)
    nc,cc=chemistry(candidate,True)
    nh,ch=chemistry(historic,False)
    return pd.DataFrame({"same_sample":names.to_numpy(),"core_match":key_match(candidate)&key_match(historic),
        "candidate_shared":nc,"historical_shared":nh,"candidate_chemistry":cc,"historical_chemistry":ch,
        "accepted":names.to_numpy()&key_match(candidate)&key_match(historic)&cc&ch},index=raw.index)


def run():
    folder=RESULTS_DIR/"model_1/expansion"/EXPANSION
    if file_sha256(folder/"manifest.json")!=EXPANSION_SHA:
        raise ValueError("Expansion manifest differs")
    follow=folder/"provenance_followup"
    if file_sha256(follow/"manifest.json")!=PROVENANCE_SHA:
        raise ValueError("Provenance manifest differs")
    evidence,audit,features,_,links=load_evidence(folder)
    source=local_training_source("model_1")
    digest=file_sha256(source)
    if digest!=evidence["source_sha256"]:
        raise ValueError("Historical source differs")
    prepared,keep,domain=cohort(source)
    x,y,groups=(v.loc[keep] for v in (prepared.x,prepared.y,prepared.groups))
    if len(prepared.x)!=3420 or len(x)!=3263:
        raise ValueError("Cohort differs")
    splits=list(KFold(5,shuffle=True,random_state=SEED).split(x))
    check_pilot(RESULTS_DIR/"generalization/model_1_oasis_like"/PILOT,x,y,groups,splits,digest,domain)
    if not np.array_equal(prepared.raw_frame.index,np.arange(len(prepared.raw_frame))):
        raise ValueError("Raw historical row positions differ")
    matches=pd.read_csv(follow/"historical_matches.csv",low_memory=False)
    selected,ambiguous=one_to_one(matches)
    selected=selected.merge(audit[["candidate_id","source_dataset","source_file","source_row_position",
        "raw_source_sha256","source_sample"]],on="candidate_id",validate="one_to_one")
    if not selected.base_row_position.isin(prepared.x.index).all() or not selected.source_dataset.eq(ANDEAN).all():
        raise ValueError("Audited identity differs")
    if selected.duplicated(["source_file","source_row_position"]).any():
        raise ValueError("Raw positions reused")
    candidate=features.set_index("candidate_id")
    recovered=pd.DataFrame(index=prepared.x.index.astype(str))
    recovered.index.name="row_id"
    checks=[]
    inventory=[]
    for item in read(folder/"raw_metadata_inventory.json"):
        if item["source"]!=ANDEAN:
            continue
        name=item["file"]
        if Path(name).name!=name:
            raise ValueError("Invalid partition")
        path=DATA_DIR/"raw/model_1/research"/ANDEAN/name
        sha=file_sha256(path)
        if sha!=item["sha256"]:
            raise ValueError("Raw partition differs")
        part=selected.loc[selected.source_file.eq(name)].copy()
        if not part.raw_source_sha256.eq(sha).all():
            raise ValueError("Candidate source hash differs")
        enc=detect_georoc_encoding(path)
        header=list(pd.read_csv(path,nrows=0,encoding=enc).columns)
        mapping=infer_georoc_column_map(header)
        extras=measurement_map(header)
        raw=pd.read_csv(path,usecols=list(dict.fromkeys([*mapping,*[v[0] for v in extras.values()]])),
            encoding=enc,dtype=str,low_memory=False)
        pos=pd.to_numeric(part.source_row_position,errors="raise")
        if pos.mod(1).ne(0).any() or pos.lt(0).any() or pos.ge(len(raw)).any():
            raise ValueError("Raw position out of range")
        pos=pos.astype(int)
        subset=raw.loc[pos.to_numpy(),list(mapping)]
        hr,_=harmonize_chunk(subset,column_map=mapping,source_name=ANDEAN,source_file=name,
            allowed_material_types=("WHOLE ROCK",),minimum_predictors=8,iron_total_mode="original")
        part=part.set_index(pos.to_numpy())
        if not hr.index.is_unique:
            raise ValueError("Raw identity duplicated")
        # Missing harmonized rows are explicitly rejected, not silently reassigned.
        hr=hr.reindex(part.index)
        cx=candidate.loc[part.candidate_id].copy().set_axis(part.index)
        cx["source_sample"]=part.source_sample
        hx=prepared.raw_frame.loc[part.base_row_position].copy().set_axis(part.index)
        result=corroborate(hr,cx,hx)
        result["row_id"]=part.base_row_position.astype(str)
        result["candidate_id"]=part.candidate_id
        result["raw_position"]=part.index
        result["file"]=name
        result["raw_sha256"]=sha
        checks.append(result.reset_index(drop=True))
        good=result.loc[result.accepted]
        for element,(column,unit) in extras.items():
            field="georoc_"+element+"_ppm"
            if field not in recovered:
                recovered[field]=np.nan
            recovered.loc[good.row_id,field]=numeric_ppm(raw.loc[good.index,column],unit).to_numpy()
        inventory.append({"file":name,"sha256":sha,"linked":len(part),"accepted":len(good),
            "measurement_headers":extras,"raw_observed":{element:int(numeric_ppm(raw[column],unit).notna().sum())
                for element,(column,unit) in extras.items()}})
        if file_sha256(path)!=sha:
            raise ValueError("Raw partition changed")
    check=pd.concat(checks,ignore_index=True)
    if len(check)!=len(selected) or not check.row_id.is_unique:
        raise ValueError("Recovery link coverage differs")
    academic=recovered.loc[x.index.astype(str)].set_axis(x.index)
    records=[]
    by_band=[]
    bands=pd.cut(y,[0,5,10,20,40,60,80,100,np.inf])
    for name in academic:
        records.append({"feature":name,**profile(academic[name],y,groups)})
        for band,idx in bands.groupby(bands,observed=True).groups.items():
            by_band.append({"feature":name,"band":str(band),**profile(academic.loc[idx,name],y.loc[idx],groups.loc[idx])})
    recovered.insert(0,"academic_member",recovered.index.isin(x.index.astype(str)))
    out=RESULTS_DIR/"generalization/model_1_expansion_linked_recovery"/(
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")+"_"+uuid4().hex[:8])
    out.mkdir(parents=True,exist_ok=False)
    tables={"recovered_measurements.csv":recovered.reset_index(),"row_checks.csv":check,
        "ambiguous_links.csv":ambiguous,"feature_profiles.csv":pd.DataFrame(records),
        "coverage_by_band.csv":pd.DataFrame(by_band)}
    for name,table in tables.items():
        write_table(out/name,table)
    write_legacy_evidence_json(out/"raw_inventory.json",inventory)
    if file_sha256(source)!=digest:
        raise ValueError("Historical source changed")
    write_legacy_evidence_json(out/"manifest.json",{**links,"state":"review_required","scope":"linked_measurement_recovery",
        "source_sha256":digest,"historical_rows":len(prepared.x),"academic_rows":len(x),"linked":len(check),
        "accepted":int(check.accepted.sum()),"ambiguous_edges":len(ambiguous),
        "measurement_columns":list(academic.columns),"nonempty_measurements":int(academic.notna().sum().sum()),
        "training":False,"acquisition":False,"methods_certified":False,"production_schema_changed":False,
        "outputs":{name:file_sha256(out/name) for name in [*tables,"raw_inventory.json"]}})
    print(pd.DataFrame(inventory).drop(columns=["measurement_headers","raw_observed"]).to_string(index=False))
    print(pd.DataFrame(records).to_string(index=False))
    print(f"Estado: review_required | Evidencias: {out}",flush=True)
    return out


if __name__=="__main__":
    run()
