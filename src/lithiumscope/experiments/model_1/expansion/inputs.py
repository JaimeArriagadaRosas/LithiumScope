"""Bind the candidate views to the frozen source and original local partitions."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.paths import DATA_DIR, RESULTS_DIR
from lithiumscope.datasets.management.local_sources import local_training_source
from lithiumscope.datasets.management.research_sources import M1_RESEARCH_SOURCES
from lithiumscope.experiments.data_lab.candidate_audit import FOLD_SHA256, REFERENCE
from lithiumscope.results.eda_artifacts import verify_eda
from lithiumscope.datasets.readers.georoc_raw_reader import detect_georoc_encoding

METHOD_NAMES = {"limethod", "limethodppm", "lippmmethod", "lianalyticalmethod", "limethodofanalysis"}
CRS_NAMES = {"coordinatecrs", "coordinate referencesystem", "coordinatesystem", "crs",
             "geodeticdatum", "coordinatedatum", "coordinatereferencesystem"}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _column(header, names):
    found = [name for name in header if re.sub(r"[^a-z0-9]", "", name.casefold()) in names]
    return found[0] if len(found) == 1 else None


def _raw_metadata(spec, frame, manifest, raw_root):
    """Read exact original headers; never infer method from Li(PPM) or CRS from bounds."""
    records = []
    frame = frame.copy()
    frame["raw_source_sha256"] = ""
    for name, rows in frame.groupby("source_file", sort=True):
        if Path(name).name != name or name not in manifest["source_sha256"]:
            raise ValueError("Candidate partition not bound to EDA source")
        path = raw_root/spec.key/name
        expected = manifest["source_sha256"][name]
        frame.loc[rows.index, "raw_source_sha256"] = expected
        info = {"source":spec.key,"file":name,"sha256":expected,"candidate_rows":len(rows),
                "raw_state":"missing","method_column":None,"crs_column":None}
        if not path.is_file():
            records.append(info)
            continue
        if file_sha256(path) != expected:
            raise ValueError("Raw GEOROC partition differs from candidate EDA: "+name)
        encoding = detect_georoc_encoding(path)
        header = list(pd.read_csv(path, nrows=0, encoding=encoding).columns)
        method, crs = _column(header, METHOD_NAMES), _column(header, CRS_NAMES)
        info.update(raw_state="verified",method_column=method,crs_column=crs,
                    metadata_header_candidates=[h for h in header if re.search(
                        r"method|technique|instrument|datum|crs|coordinate|reference|citation",h,re.I)])
        columns = list(dict.fromkeys(c for c in (method,crs) if c))
        if columns:
            raw = pd.read_csv(path,usecols=columns,encoding=encoding,dtype=str,low_memory=False)
            positions = pd.to_numeric(rows.source_row_position,errors="raise")
            if (positions.lt(0).any() or positions.ge(len(raw)).any()
                    or positions.mod(1).ne(0).any()):
                raise ValueError("Candidate raw positions out of range")
            for column,target in ((method,"source_li_method"),(crs,"source_coordinate_crs")):
                if column:
                    values = raw[column].iloc[positions.astype(int).to_numpy()].fillna("").str.strip()
                    frame.loc[rows.index,target] = values.to_numpy()
        if file_sha256(path) != expected:
            raise ValueError("Raw partition changed during metadata audit")
        records.append(info)
    return frame,records


def load_inputs(*, results_root=RESULTS_DIR, raw_root=None, source=None):
    root=Path(results_root)
    raw_root=Path(raw_root) if raw_root is not None else DATA_DIR/"raw/model_1/research"
    source=Path(source) if source is not None else local_training_source("model_1")
    digest=file_sha256(source)
    folds_path=root/"model_1/runs"/REFERENCE/"tables/outer_fold_manifest.csv"
    if file_sha256(folds_path)!=FOLD_SHA256:
        raise ValueError("Historical fold manifest differs")
    folds=pd.read_csv(folds_path,dtype={"row_index":str}).sort_values("row_position")
    if (len(folds)!=3420 or folds.spatial_group.nunique()!=244
            or set(folds.outer_fold)!=set(range(1,6))):
        raise ValueError("Frozen historical cohort differs")
    frames, inputs, inventory = [], [], []
    for spec in M1_RESEARCH_SOURCES:
        folder=root/"model_1/candidate_eda"/spec.key
        for run in sorted(folder.glob("eda_*"),reverse=True):
            try:
                m=read_json(run/"manifest.json")
                if m.get("source_key")!=spec.key or m.get("base_sha256")!=digest or not verify_eda(run,m["signature"]):
                    continue
                path=run/"tables/candidate_view.csv"
                frame=pd.read_csv(path,low_memory=False)
                if not frame.source_dataset.eq(spec.key).all():
                    raise ValueError("Candidate view identity differs")
            except (OSError,ValueError,KeyError,TypeError):
                continue
            frame,raw=_raw_metadata(spec,frame,m,raw_root)
            frames.append(frame)
            inventory.extend(raw)
            inputs.append({"source":spec.key,"doi":spec.doi,"run_id":run.name,
                           "view_sha256":file_sha256(path),"manifest_sha256":file_sha256(run/"manifest.json")})
            break
        else:
            raise FileNotFoundError("No intact local EDA for "+spec.key+"; acquisition disabled")
    base=pd.read_csv(source,low_memory=False)
    if file_sha256(source)!=digest:
        raise ValueError("Historical source changed")
    return source,base,folds,pd.concat(frames,ignore_index=True),inputs,inventory
