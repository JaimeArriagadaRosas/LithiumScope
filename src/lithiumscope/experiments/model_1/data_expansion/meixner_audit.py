"""Meixner A1 sample audit and target-independent candidate identity search.

No joins admitted to training; name/coordinates/silica are review evidence only.
"""
from datetime import datetime, timezone
from uuid import uuid4
from pathlib import Path
import numpy as np
import pandas as pd
from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.paths import DATA_DIR, RESULTS_DIR
from lithiumscope.datasets.management.sources.local import local_training_source
from lithiumscope.datasets.adapters.georoc.schema import infer_georoc_column_map
from lithiumscope.datasets.readers.georoc_raw_reader import detect_georoc_encoding
from lithiumscope.experiments.data_lab.source_studies.geochemistry import raw_view
from lithiumscope.experiments.data_lab.source_studies.meixner import ORIGINALS, parse_a1, match_names
from lithiumscope.experiments.data_lab.screening.candidates import CHEM, sample, num
from lithiumscope.experiments.model_1.protocols.cohorts import cohort
from lithiumscope.experiments.model_1.data_expansion.raw_recovery import EXPANSION, EXPANSION_SHA
from lithiumscope.experiments.model_1.data_expansion.sensitivity_data import load_evidence, read
from lithiumscope.experiments.data_lab.source_studies.directed_intake import ROOT, MEIXNER
from lithiumscope.results.atomic_exports import write_table, write_legacy_evidence_json

def run():
    folder=ROOT/MEIXNER
    for name,digest in ORIGINALS.items():
        if file_sha256(folder/name)!=digest:
            raise ValueError("Meixner original differs: "+name)
    table=pd.read_excel(folder/"126_2019_915_MOESM2_ESM.xls",sheet_name="Sheet1",header=None,engine="xlrd")
    rows=parse_a1(table)
    query=rows.loc[~rows.repeat&rows.Li_ppm.notna()].copy()
    source=local_training_source("model_1")
    digest=file_sha256(source)
    expansion=RESULTS_DIR/"model_1/expansion"/EXPANSION
    if file_sha256(expansion/"manifest.json")!=EXPANSION_SHA:
        raise ValueError("Expansion manifest differs")
    evidence,_,_,_,links=load_evidence(expansion)
    if evidence["source_sha256"]!=digest:
        raise ValueError("Historical source differs")
    prepared,_,_=cohort(source)
    historical=prepared.raw_frame
    matches=match_names(query,historical,"historical_raw",source.name)
    names=set(sample(query))
    inventory=[]
    for item in read(expansion/"raw_metadata_inventory.json"):
        key,name=item["source"],item["file"]
        if Path(name).name!=name or Path(key).name!=key:
            raise ValueError("Invalid raw path")
        path=DATA_DIR/"raw/model_1/research"/key/name
        sha=file_sha256(path)
        if sha!=item["sha256"]:
            raise ValueError("Research partition differs")
        encoding=detect_georoc_encoding(path)
        mapping=infer_georoc_column_map(pd.read_csv(path,nrows=0,encoding=encoding).columns)
        count=0
        if "source_sample" in mapping.values():
            for chunk in pd.read_csv(path,usecols=list(mapping),dtype=str,encoding=encoding,chunksize=10000):
                normalized=chunk.rename(columns=mapping)
                subset=chunk.loc[sample(normalized).isin(names)]
                if not subset.empty:
                    count+=len(subset)
                    matches.extend(match_names(query,raw_view(subset,mapping),key,name))
        inventory.append({"source":key,"file":name,"sha256":sha,"name_matches_raw_rows":count,
                          "sample_column_available":"source_sample" in mapping.values()})
        if file_sha256(path)!=sha:
            raise ValueError("Research partition changed")
    edges=pd.DataFrame(matches,columns=["excel_row","source_sample","candidate_source","candidate_file",
        "candidate_position","candidate_sample","candidate_publication","candidate_material","candidate_rock",
        "site_match","silica_match","chemical_predictors","oxide_sum","qc_diagnostic","candidate_has_li",
        "identity_review_candidate","approved_for_training"])
    review=edges.loc[edges.identity_review_candidate.eq(True)]
    summary=query.copy()
    summary["name_matches"]=summary.excel_row.map(edges.groupby("excel_row").size()).fillna(0).astype(int)
    summary["site_silica_matches"]=summary.excel_row.map(review.groupby("excel_row").size()).fillna(0).astype(int)
    summary["matches_with_8_chem"]=summary.excel_row.map(review.loc[review.chemical_predictors.ge(8)].groupby("excel_row").size()).fillna(0).astype(int)
    summary["historical_name_match"]=summary.excel_row.isin(edges.loc[edges.candidate_source.eq("historical_raw"),"excel_row"])
    summary["prepared_historical_name_match"]=summary.excel_row.isin(edges.loc[
        edges.candidate_source.eq("historical_raw")&edges.candidate_position.isin(prepared.x.index.astype(str)),"excel_row"])
    out=RESULTS_DIR/"generalization/model_1_meixner_identity_audit"/(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")+"_"+uuid4().hex[:8])
    out.mkdir(parents=True,exist_ok=False)
    tables={"a1_inventory.csv":rows,"sample_review.csv":summary,"identity_edges.csv":edges,
            "partition_inventory.csv":pd.DataFrame(inventory)}
    for name,t in tables.items():
        write_table(out/name,t)
    if file_sha256(source)!=digest or any(file_sha256(folder/n)!=h for n,h in ORIGINALS.items()):
        raise ValueError("Original source changed")
    write_legacy_evidence_json(out/"manifest.json",{**links,"state":"review_required","source_sha256":digest,
        "original_hashes":ORIGINALS,"primary_li_rows":len(query),"name_matched_samples":int(summary.name_matches.gt(0).sum()),
        "site_silica_matched_samples":int(summary.site_silica_matches.gt(0).sum()),
        "matched_with_8_chem_samples":int(summary.matches_with_8_chem.gt(0).sum()),
        "target_used_for_matching":False,"new_rows_approved":0,"training":False,
        "outputs":{name:file_sha256(out/name) for name in tables}})
    print(summary.groupby("rock_section")[["name_matches","site_silica_matches","matches_with_8_chem"]].sum().to_string())
    print(f"Estado: review_required | Evidencias: {out}",flush=True)
    return out


if __name__=="__main__":
    run()
