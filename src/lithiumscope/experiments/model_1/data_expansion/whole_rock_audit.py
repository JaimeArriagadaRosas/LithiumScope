"""Directed whole-rock source audit; immutable originals, target-free duplicate flags.

Outputs are research review tables, never training inputs.
"""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4
import numpy as np
import pandas as pd

from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.paths import RESULTS_DIR
from lithiumscope.datasets.management.sources.local import local_training_source
from lithiumscope.experiments.data_lab.source_studies.whole_rock import FILES, META, PUNA, parse_tables, duplicate_edges, summarize
from lithiumscope.experiments.model_1.protocols.cohorts import cohort
from lithiumscope.experiments.model_1.data_expansion.raw_recovery import EXPANSION, EXPANSION_SHA
from lithiumscope.experiments.model_1.data_expansion.sensitivity_data import load_evidence, historical_tokens
from lithiumscope.experiments.data_lab.source_studies.directed_intake import ROOT
from lithiumscope.results.atomic_exports import write_table, write_legacy_evidence_json

def run():
    records=[]; inventories=[]; censored=[]; original_hashes={}
    for key,(name,expected) in FILES.items():
        path=ROOT/key/name
        if file_sha256(path)!=expected:
            raise ValueError("Original hash differs: "+name)
        original_hashes[str(path.relative_to(ROOT))]=expected
        puna=key==PUNA
        with pd.ExcelFile(path,engine="openpyxl") as book:
            samples=book.parse("1 Samples" if puna else "2 Samples",header=None)
            data=book.parse("2 Data" if puna else "3 Data",header=None)
            frame,inventory,flags=parse_tables(samples,data,key)
        records.append(frame); inventories.append(inventory); censored.extend(flags)
    name,expected=META
    path=ROOT/PUNA/name
    if file_sha256(path)!=expected:
        raise ValueError("Puna metadata differs")
    original_hashes[str(path.relative_to(ROOT))]=expected
    query=pd.concat(records,ignore_index=True)
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
    candidate=features.copy()
    for c in ("source_sample","source_unique_id","source_publication"):
        if c in audit:
            candidate[c]=audit[c].to_numpy()
    edges=duplicate_edges(query,hist,"historical_raw")+duplicate_edges(query,candidate,"prior_expansion")
    for key,part in query.groupby("source_dataset"):
        other=query.loc[query.source_dataset.ne(key)]
        edges.extend(duplicate_edges(part,other,"directed_other_source"))
        for idx in part.index:
            edges.extend(duplicate_edges(part.loc[[idx]],part.drop(index=idx),"directed_same_source"))
    known=set().union(*historical_tokens(hist),*pubs)
    review=summarize(query,edges,known,domain,set(prepared.groups))
    report=review.groupby("source_dataset")[["positive_li","basic_qc","academic_range","possible_duplicate",
        "publication_overlap","academic_review_candidate","spatial_review_candidate"]].sum().reset_index()
    report.insert(1,"rows",report.source_dataset.map(review.source_dataset.value_counts()))
    bands=review.assign(band=pd.cut(review.Li_icpms,[0,5,10,20,40,60,80,100,126.12,696,np.inf]))
    bands=bands.groupby(["source_dataset","band"],observed=True).size().rename("rows").reset_index()
    out=RESULTS_DIR/"generalization/model_1_directed_whole_rock_audit"/(
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")+"_"+uuid4().hex[:8])
    out.mkdir(parents=True,exist_ok=False)
    tables={"row_review.csv":review,"summary.csv":report,"band_counts.csv":bands,
        "measurement_inventory.csv":pd.concat(inventories,ignore_index=True),
        "duplicate_edges.csv":pd.DataFrame(edges,columns=["source_dataset","source_sample","comparison","comparison_position",
            "name_match","id_match","site_match","shared_chemistry","chemistry_agreement","possible_duplicate","comparison_sample"]),
        "censoring.csv":pd.DataFrame(censored,columns=["source_dataset","source_sample","parameter","raw","kind","limit","value"])}
    for name,t in tables.items():
        write_table(out/name,t)
    if file_sha256(source)!=digest or any(file_sha256(ROOT/p)!=v for p,v in original_hashes.items()):
        raise ValueError("Source changed during audit")
    write_legacy_evidence_json(out/"manifest.json",{**links,"state":"review_required","source_sha256":digest,
        "original_hashes":original_hashes,"rows":len(review),"domain":domain,"new_rows_approved":0,
        "target_used_for_matching":False,"training":False,"production_schema_changed":False,
        "independent_holdout_certified":False,"duplicate_scope":"historical_raw_and_prior_candidate_inventory_not_all_raw_partitions",
        "canonical_columns_are_review_aliases":True,"outputs":{name:file_sha256(out/name) for name in tables}})
    print(report.to_string(index=False))
    print(f"Estado: review_required | Evidencias: {out}",flush=True)
    return out


if __name__=="__main__":
    run()
