"""Verified bibliography and conservative research-only eligibility for expansion."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd
import numpy as np

from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.paths import RESULTS_DIR, DATA_DIR
from lithiumscope.experiments.model_1.expansion.cohorts import GOOD_STATES, POOLS, train_domain_fraction
from lithiumscope.experiments.model_1.expansion.identity import multivariate_support

DOI = re.compile(r"10\.\d{4,9}/[^\s<>]+", re.I)
REF = re.compile(r"^\[(\d+)\]\s*(.+)")
ID = re.compile(r"\[(\d+)\]")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def tokens(value):
    value=str(value).strip().casefold()
    if value in {"","nan","none","unknown","<na>"}:
        return set()
    out={"text:"+re.sub(r"\s+"," ",value)}
    out.update("doi:"+v.rstrip(".,;").lower() for v in DOI.findall(value))
    if re.fullmatch(r"(?:\s*\[\d+\][,; ]*)+",value):
        out.update("georoc:"+v for v in ID.findall(value))
    return out


def bibliography(footer):
    index={}
    for row in footer:
        if row["section"]!="references":
            continue
        for cell in row["cells"]:
            match=REF.match(cell.strip())
            if match:
                key=(row["source"],row["file"],match[1])
                index.setdefault(key,set()).add(match[2].strip())
    return index


def candidate_tokens(row, index):
    ids=ID.findall(str(row.source_publication))
    if not ids:
        return set()
    refs=[]
    for identifier in ids:
        values=index.get((row.source_dataset,row.source_file,identifier),set())
        if len(values)!=1:
            return set()
        refs.append(next(iter(values)))
    return {"georoc:"+i for i in ids}.union(*(tokens(v) for v in refs))


def historical_tokens(frame):
    columns=[c for c in ("source_publication","Geochemistry Reference") if c in frame]
    result=[]
    for _,row in frame[columns].iterrows():
        result.append(set().union(*(tokens(row[c]) for c in columns)))
    return result


def load_evidence(run):
    folder=Path(run).resolve()
    if folder.parent!=(RESULTS_DIR/"model_1/expansion").resolve():
        raise ValueError("Expected existing expansion run")
    m=read(folder/"manifest.json")
    p=folder/"provenance_followup"
    follow=read(p/"manifest.json")
    if (m.get("scope")!="M1_expansion_preparation_not_training"
            or follow.get("expansion_manifest_sha256")!=file_sha256(folder/"manifest.json")
            or follow.get("source_sha256")!=m.get("source_sha256")
            or follow.get("training") is not False):
        raise ValueError("Expansion/provenance identity mismatch")
    for root,manifest,names in ((folder,m,("row_audit.csv","candidate_features.csv","raw_metadata_inventory.json")),
          (p,follow,("bibliography_footer.json","historical_matches.csv","partition_status.json"))):
        for name in names:
            if file_sha256(root/name)!=manifest["outputs"][name]:
                raise ValueError("Evidence hash differs: "+name)
    raw=read(folder/"raw_metadata_inventory.json")
    status={r["file"]:r["sha256"] for r in read(p/"partition_status.json")}
    for row in raw:
        name,key=row["file"],row["source"]
        if Path(name).name!=name or Path(key).name!=key or row["raw_state"]!="verified":
            raise ValueError("Raw partition unverified")
        if status.get(name)!=row["sha256"] or file_sha256(DATA_DIR/"raw/model_1/research"/key/name)!=row["sha256"]:
            raise ValueError("Raw partition hash differs")
    audit=pd.read_csv(folder/"row_audit.csv",low_memory=False)
    features=pd.read_csv(folder/"candidate_features.csv",low_memory=False)
    if not audit.candidate_id.is_unique or not audit.candidate_id.equals(features.candidate_id):
        raise ValueError("Candidate identity differs")
    for name in ("Li_icpms","spatial_group","source_dataset"):
        if not audit[name].equals(features[name]):
            raise ValueError("Candidate label/group/source differs")
    historical=set(pd.read_csv(p/"historical_matches.csv").candidate_id)
    refs=bibliography(read(p/"bibliography_footer.json"))
    publications=[candidate_tokens(row,refs) for row in audit.itertuples(index=False)]
    audit["bibliography_resolved"]=[bool(v) for v in publications]
    audit["historical_exact_overlap"]=audit.candidate_id.isin(historical)
    audit["sensitivity_eligible"]=(audit.screening_state.isin(GOOD_STATES)
         & audit.raw_verified.eq(True) & audit.prepared_coordinates_valid.eq(True)
         & audit.bibliography_resolved & ~audit.historical_exact_overlap
         & audit.quality_ok.eq(True) & audit.material_ok.eq(True)
         & audit.lithium_unit_and_raw_ok.eq(True)
         & ~audit.exact_copy.eq(True) & ~audit.review_decision.eq("reject"))
    return m,audit,features,publications,{"expansion_manifest_sha256":file_sha256(folder/"manifest.json"),
                   "provenance_manifest_sha256":file_sha256(p/"manifest.json")}


def reserve(audit, publications, groups, historical_refs):
    """Reserve new groups by fixed hash, including a separate high-tail stratum."""
    eligible=audit.sensitivity_eligible.to_numpy(bool)
    known=set().union(*historical_refs)
    new=eligible & ~audit.spatial_group.isin(set(groups)).to_numpy()
    new &= np.array([not bool(v&known) for v in publications])
    held_groups=set()
    for state in sorted(GOOD_STATES):
        g=set(audit.loc[new & audit.screening_state.eq(state),"spatial_group"])
        ordered=sorted(g,key=lambda v:hashlib.sha256(("M1-sensitivity-v1|"+v).encode()).hexdigest())
        if len(ordered)>=5:
            held_groups.update(ordered[:max(1,len(ordered)//5)])
    held=new & audit.spatial_group.isin(held_groups).to_numpy()
    reserved_refs=set().union(*(publications[i] for i in np.flatnonzero(held)))
    linked=np.array([bool(v&reserved_refs) for v in publications])
    pool=eligible & ~held & ~linked & ~audit.spatial_group.isin(held_groups).to_numpy()
    return held,pool


def fold_pools(audit, cx, train_x, numeric, pool, publications, blocked_groups, blocked_refs):
    safe=pool & ~audit.spatial_group.isin(blocked_groups).to_numpy()
    safe &= np.array([not bool(v&blocked_refs) for v in publications])
    other=safe & audit.screening_state.eq("other_domain_review").to_numpy()
    support=np.zeros(len(audit),dtype=bool)
    if other.any():
        sub=cx.loc[other]
        support[other]=(multivariate_support(sub,train_x,numeric).chemistry_nn_supported
                        & train_domain_fraction(sub,train_x,numeric).le(.25)).to_numpy()
    return {name:np.flatnonzero(safe & audit.screening_state.isin(states).to_numpy()
                    & (~other | support)) for name,states in POOLS.items()}
