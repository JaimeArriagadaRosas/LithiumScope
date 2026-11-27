"""Chen corrected chemistry, bibliography and target-free overlap screen.

Compilation is not an independent holdout; no row is approved for training.
"""
from datetime import datetime, timezone
from uuid import uuid4
import re
import numpy as np
import pandas as pd
from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.paths import RESULTS_DIR
from lithiumscope.datasets.management.sources.local import local_training_source
from lithiumscope.experiments.data_lab.candidate_screening import CHEM, num, group
from lithiumscope.experiments.model_1.balanced.academic import cohort
from lithiumscope.experiments.model_1.expansion.raw_recovery import EXPANSION, EXPANSION_SHA
from lithiumscope.experiments.model_1.expansion.sensitivity_data import load_evidence, historical_tokens, tokens
from lithiumscope.experiments.model_1.augmentation.chen_profile import ROOT,SOURCE,NAME,SHA,SHEET,profile
from lithiumscope.results.atomic_exports import write_table, write_legacy_evidence_json

ID=re.compile(r"\[(\d+)\]")
REF=re.compile(r"^\s*\[(\d+)\]\s+(.+)$")
FIELDS=[c for c in CHEM if c!="Fe2O3"]
OX=[c for c in FIELDS if "_icpms" not in c]


def bibliography(table):
    index={}
    for i,value in table.iloc[2:,0].items():
        match=REF.match(str(value))
        if match:
            index.setdefault(match[1],set()).add(match[2].strip())
    return index


def publication_sets(values,index):
    resolved=[]; pubsets=[]
    for value in values:
        ids=ID.findall(str(value))
        okay=bool(ids) and all(len(index.get(i,set()))==1 for i in ids)
        p={"georoc:"+i for i in ids}
        if okay:
            for i in ids:
                p.update(tokens(next(iter(index[i]))))
        resolved.append(okay)
        pubsets.append(p)
    return np.asarray(resolved,bool),pubsets


def bounds(frame):
    xy=num(frame,["Latitude","Longitude"])
    ends=num(frame,["Latitude_min","Latitude_max","Longitude_min","Longitude_max"])
    for axis in ("Latitude","Longitude"):
        for suffix in ("_min","_max"):
            ends[axis+suffix]=ends[axis+suffix].fillna(xy[axis])
    # Reject inverted bounds rather than silently swapping endpoints.
    valid=(ends.Latitude_min.between(-90,90)&ends.Latitude_max.between(-90,90)&
           ends.Longitude_min.between(-180,180)&ends.Longitude_max.between(-180,180)&
           ends.Latitude_min.le(ends.Latitude_max)&ends.Longitude_min.le(ends.Longitude_max))
    return ends.to_numpy(float),valid.to_numpy(bool)


def overlap_edges(query,base,query_pubs,base_pubs,origin,within=False):
    """Broad location overlap or shared reference, then non-Li chemistry agreement."""
    a=num(query,FIELDS).to_numpy(float)
    b=num(base,FIELDS).to_numpy(float)
    qa,qv=bounds(query); ba,bv=bounds(base)
    # Match on nine major oxides excluding Fe; retain trace corroboration.
    major=np.array([c in OX for c in FIELDS])
    ref_positions={}
    for j,values in enumerate(base_pubs):
        for v in values:
            ref_positions.setdefault(v,[]).append(j)
    edges=[]
    for i in range(len(query)):
        site=qv[i]&bv&(ba[:,0]<=qa[i,1]+.001)&(ba[:,1]>=qa[i,0]-.001)&(
             ba[:,2]<=qa[i,3]+.001)&(ba[:,3]>=qa[i,2]-.001)
        pub=np.zeros(len(base),bool)
        for value in query_pubs[i]:
            pub[ref_positions.get(value,[])]=True
        ix=np.flatnonzero(site|pub)
        if within:
            ix=ix[ix>i]
        if not len(ix):
            continue
        other=b[ix]; shared=np.isfinite(other)&np.isfinite(a[i])
        agree=(np.abs(other-a[i])<=np.maximum(.001,.02*np.maximum(np.abs(other),np.abs(a[i]))))|~shared
        n=shared.sum(axis=1); nmajor=(shared&major).sum(axis=1)
        # Strong chemical overlap: at least 8 independent measurements,
        # at least 5 major oxides and all shared values in tolerance.
        good=(n>=8)&(nmajor>=5)&agree.all(axis=1)
        for k in np.flatnonzero(good):
            j=ix[k]
            edges.append({"excel_row":int(query.iloc[i].excel_row),"comparison":origin,
                "comparison_position":str(base.index[j]),
                "comparison_excel_row":int(base.iloc[j].excel_row) if within else None,
                "site_overlap":bool(site[j]),"publication_overlap":bool(pub[j]),
                "shared_chemistry":int(n[k]),"shared_major_oxides":int(nmajor[k])})
    return edges


def run():
    original=ROOT/SOURCE/NAME
    if file_sha256(original)!=SHA:
        raise ValueError("Chen original differs")
    table=pd.read_excel(original,sheet_name=SHEET,header=None,engine="openpyxl",dtype=object)
    if table.shape!=(50145,66):
        raise ValueError("Chen layout differs")
    rows,inventory,bands,summary=profile(table)
    if len(rows)!=3680 or "FeOT_total_as_FeO" not in rows or not rows.excel_row.is_unique:
        raise ValueError("Chen observation inventory differs")
    index=bibliography(table)
    resolved,qp=publication_sets(rows.source_publication,index)
    source=local_training_source("model_1")
    digest=file_sha256(source)
    expansion=RESULTS_DIR/"model_1/expansion"/EXPANSION
    if file_sha256(expansion/"manifest.json")!=EXPANSION_SHA:
        raise ValueError("Expansion manifest differs")
    evidence,audit,features,pubs,links=load_evidence(expansion)
    if evidence["source_sha256"]!=digest:
        raise ValueError("Historical source differs")
    prepared,keep,domain=cohort(source)
    if len(prepared.x)!=3420 or int(keep.sum())!=3263:
        raise ValueError("Frozen cohort differs")
    hist=prepared.raw_frame
    hp=historical_tokens(hist)
    edges=overlap_edges(rows,hist,qp,hp,"historical_raw")
    edges+=overlap_edges(rows,features,qp,pubs,"prior_expansion")
    edges+=overlap_edges(rows,rows,qp,qp,"chen_internal",within=True)
    blocked={r["excel_row"] for r in edges}
    blocked.update(r["comparison_excel_row"] for r in edges if r["comparison_excel_row"] is not None)
    # Flag exact fingerprints independently of publication and coordinate range.
    fingerprint=rows[["Latitude","Longitude",*FIELDS]]
    exact=fingerprint.duplicated(keep=False)&fingerprint[FIELDS].notna().sum(axis=1).ge(8)
    rows["exact_internal_fingerprint"]=exact
    rows["possible_duplicate"]=rows.excel_row.isin(blocked)|exact
    rows["bibliography_resolved"]=resolved
    known_h=set().union(*hp)
    known_c=set().union(*pubs)
    rows["historical_publication_overlap"]=[bool(p&known_h) for p in qp]
    rows["candidate_publication_overlap"]=[bool(p&known_c) for p in qp]
    rows["existing_spatial_group"]=rows.spatial_group.isin(set(prepared.groups))
    # Report Andean versus other arcs without fabricating geological categories.
    rows["arc_label"]=rows.source_location.fillna("").astype(str).str.split(" / ").str[0]
    rows["academic_range"]=rows.Li_icpms.between(domain["q025"],domain["q975"])
    rows["academic_tail"]=rows.academic_range&rows.Li_icpms.gt(40)
    rows["transfer_review_candidate"]=rows.basic_qc&rows.academic_range&resolved&~rows.possible_duplicate&~(
        rows.historical_publication_overlap|rows.candidate_publication_overlap)
    rows["approved_for_training"]=False
    flags=["basic_qc","bibliography_resolved","possible_duplicate","historical_publication_overlap",
        "candidate_publication_overlap","existing_spatial_group","academic_tail","transfer_review_candidate"]
    arc=rows.groupby("arc_label",dropna=False)[flags].sum().reset_index()
    arc.insert(1,"rows",arc.arc_label.map(rows.arc_label.value_counts()))
    tail=rows.loc[rows.academic_tail]
    summary.update({"positive_rows":len(rows),"publication_ids":len(index),
        **{k:int(rows[k].sum()) for k in flags},
        "academic_tail_transfer_candidates":int(tail.transfer_review_candidate.sum()),
        "academic_tail_basic_qc":int(tail.basic_qc.sum()),
        "academic_tail_duplicates":int(tail.possible_duplicate.sum())})
    out=RESULTS_DIR/"generalization/model_1_chen_overlap_audit"/(
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")+"_"+uuid4().hex[:8])
    out.mkdir(parents=True,exist_ok=False)
    tables={"row_review.csv":rows.reset_index(drop=True),"column_inventory.csv":inventory,
        "band_counts.csv":bands,"arc_summary.csv":arc,
        "overlap_edges.csv":pd.DataFrame(edges,columns=["excel_row","comparison","comparison_position",
            "comparison_excel_row","site_overlap","publication_overlap","shared_chemistry","shared_major_oxides"]),
        "bibliography.csv":pd.DataFrame([{"reference_id":i,"reference":v,"ambiguous":len(values)!=1}
            for i,values in sorted(index.items()) for v in sorted(values)])}
    for name,t in tables.items():
        write_table(out/name,t)
    if file_sha256(original)!=SHA or file_sha256(source)!=digest:
        raise ValueError("Source changed")
    write_legacy_evidence_json(out/"manifest.json",{**links,"state":"review_required","scope":"chen_overlap_screen",
        "original_sha256":SHA,"source_sha256":digest,"summary":summary,"domain":domain,
        "target_used_for_matching":False,"new_rows_approved":0,"training":False,
        "independent_holdout_certified":False,"metadata_certified":False,
        "duplicate_scope":"historical_raw_prior_candidate_inventory_and_chen_positive_li_rows",
        "outputs":{name:file_sha256(out/name) for name in tables}})
    print(pd.Series({k:v for k,v in summary.items() if k not in ("mapped_fields","mapped_chemical_fields")}).to_string())
    print(arc.to_string(index=False))
    print(f"Estado: review_required | Evidencias: {out}",flush=True)
    return out


if __name__=="__main__":
    run()
