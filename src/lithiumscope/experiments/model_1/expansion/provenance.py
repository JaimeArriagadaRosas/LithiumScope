"""Recover local GEOROC bibliography and distinguish historical overlap from candidate duplication."""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.paths import DATA_DIR, RESULTS_DIR
from lithiumscope.datasets.management.sources.local import local_training_source
from lithiumscope.datasets.readers.georoc_raw_reader import detect_georoc_encoding
from lithiumscope.experiments.data_lab.candidate_screening import CHEM, num, sample, text
from lithiumscope.model_1.pipeline import prepare_training_data
from lithiumscope.results.atomic_exports import write_json, write_table

REF_MARKER = re.compile(r"^(references|bibliography|citations)\s*:\s*$", re.I)
ID_PATTERN = re.compile(r"\[(\d+)\]")


def read_footer(path):
    """Keep exact nonempty cells and line number; no guess about reference syntax."""
    records=[]
    section=None
    with path.open(encoding=detect_georoc_encoding(path),newline="") as handle:
        reader=csv.reader(handle)
        for row in reader:
            cells=[v.strip() for v in row if v.strip()]
            if not cells:
                continue
            if any(REF_MARKER.fullmatch(v) for v in cells):
                section="references"
            elif any(re.fullmatch(r"abbreviations\s*:\s*",v,re.I) for v in cells):
                section="abbreviations"
            if section:
                records.append({"line":reader.line_num,"section":section,"cells":cells,
                                "bracket_ids":ID_PATTERN.findall(" ".join(cells))})
    return records


def historical_matches(audit, features, base, prepared_indices):
    """Exact coordinate/Li keys are candidates for identity review, not proof."""
    def keys(frame):
        xy=num(frame,["Latitude","Longitude","Li_icpms"])
        return pd.DataFrame({"lat":xy.Latitude.round(5),"lon":xy.Longitude.round(5),
                             "li":xy.Li_icpms.round(6)},index=frame.index)
    left=keys(audit)
    left["candidate_id"]=audit.candidate_id
    left["candidate_position"]=np.arange(len(audit))
    right=keys(base)
    right["base_row_position"]=np.arange(len(base))
    right["base_prepared"]=base.index.isin(prepared_indices)
    right["base_source"]=text(base,"source_dataset")
    right["base_sample"]=sample(base)
    right["base_publication"]=text(base,"source_publication")
    right["base_unique_id"]=text(base,"source_unique_id")
    match=left.dropna(subset=["lat","lon","li"]).merge(
        right.dropna(subset=["lat","lon","li"]),on=["lat","lon","li"],how="inner")
    cpos=match.candidate_position.to_numpy(int)
    bpos=match.base_row_position.to_numpy(int)
    a=num(features,CHEM).iloc[cpos].to_numpy(float)
    b=num(base,CHEM).iloc[bpos].to_numpy(float)
    shared=np.isfinite(a)&np.isfinite(b)
    close=np.abs(a-b)<=np.maximum(.001,.02*np.maximum(np.abs(a),np.abs(b)))
    match["shared_chemistry"]=shared.sum(axis=1)
    match["chemistry_agreement"]=(shared.sum(axis=1)>=5)&(close|~shared).all(axis=1)
    match["candidate_sample"]=sample(audit).iloc[cpos].to_numpy()
    match["same_sample_name"]=(match.candidate_sample.ne("")&match.candidate_sample.eq(match.base_sample))
    match["candidate_source"]=audit.source_dataset.iloc[cpos].to_numpy()
    match["candidate_publication"]=audit.source_publication.iloc[cpos].to_numpy()
    return match


def run(expansion):
    folder=Path(expansion).resolve()
    root=(RESULTS_DIR/"model_1/expansion").resolve()
    if folder.parent!=root:
        raise ValueError("Use an existing M1 expansion run folder")
    manifest=json.loads((folder/"manifest.json").read_text(encoding="utf-8"))
    required={"row_audit.csv","candidate_features.csv","raw_metadata_inventory.json"}
    if manifest.get("scope")!="M1_expansion_preparation_not_training" or not required.issubset(manifest["outputs"]):
        raise ValueError("Not a verified M1 expansion")
    for name in required:
        if file_sha256(folder/name)!=manifest["outputs"][name]:
            raise ValueError("Expansion input changed: "+name)
    audit=pd.read_csv(folder/"row_audit.csv",low_memory=False)
    features=pd.read_csv(folder/"candidate_features.csv",low_memory=False)
    if not audit.candidate_id.is_unique or not audit.candidate_id.equals(features.candidate_id):
        raise ValueError("Candidate identities differ")
    inventory=json.loads((folder/"raw_metadata_inventory.json").read_text(encoding="utf-8"))
    footer=[]
    status=[]
    for item in inventory:
        name=item["file"]
        key=item["source"]
        if Path(name).name!=name or Path(key).name!=key:
            raise ValueError("Invalid raw partition name")
        path=DATA_DIR/"raw/model_1/research"/key/name
        if file_sha256(path)!=item["sha256"]:
            raise ValueError("Raw source changed: "+name)
        rows=read_footer(path)
        if file_sha256(path)!=item["sha256"]:
            raise ValueError("Raw source changed during scan")
        footer.extend({"source":key,"file":name,**r} for r in rows)
        status.append({"file":name,"sha256":item["sha256"],"footer_rows":len(rows),
                       "reference_rows":sum(r["section"]=="references" for r in rows)})
    source=local_training_source("model_1")
    if file_sha256(source)!=manifest["source_sha256"]:
        raise ValueError("Historical dataset changed")
    prepared=prepare_training_data(source)
    if len(prepared.x)!=manifest["historical_rows"]:
        raise ValueError("Historical prepared cohort changed")
    base=pd.read_csv(source,low_memory=False)
    matches=historical_matches(audit,features,base,prepared.frame.index)
    columns=[str(c) for c in base.columns if re.search(r"publi|refer|citat|doi|sample|source|method|datum|crs",str(c),re.I)]
    output=folder/"provenance_followup"
    output.mkdir(exist_ok=True)
    write_json(output/"bibliography_footer.json",footer)
    write_json(output/"partition_status.json",status)
    write_table(output/"historical_matches.csv",matches)
    high=audit.loc[audit.Li_icpms.gt(696),["candidate_id","source_dataset","source_publication","source_sample",
                                      "Li_icpms","screening_state","quality_ok"]].copy()
    agg=matches.groupby("candidate_id").agg(historical_matches=("base_row_position","size"),
        prepared_matches=("base_prepared","sum"),chemistry_matches=("chemistry_agreement","sum"),
        same_sample_matches=("same_sample_name","sum"))
    high=high.join(agg,on="candidate_id")
    for name in agg.columns:
        high[name]=high[name].fillna(0).astype(int)
    write_table(output/"high_lithium_overlap.csv",high)
    write_json(output/"manifest.json",{"state":"review_required","training":False,
        "source_sha256":manifest["source_sha256"],"expansion_manifest_sha256":file_sha256(folder/"manifest.json"),
        "historical_reference_columns":columns,"candidate_rows_with_exact_historical_keys":int(matches.candidate_id.nunique()),
        "outputs":{n:file_sha256(output/n) for n in ("bibliography_footer.json","partition_status.json",
                                                   "historical_matches.csv","high_lithium_overlap.csv")}})
    print(high.groupby(["source_dataset","screening_state"]).agg(rows=("candidate_id","size"),
          historical_matches=("historical_matches",lambda v:int(v.gt(0).sum())),
          prepared_matches=("prepared_matches",lambda v:int(v.gt(0).sum()))).to_string())
    print("Reference footer rows:",sum(r["section"]=="references" for r in footer))
    print("Evidence:",output)
    return output


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",required=True)
    run(parser.parse_args().run)
