"""Acquire and inventory original scientific supplements; no automatic harmonization."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import re
from uuid import uuid4
from zipfile import ZipFile, is_zipfile
import xml.etree.ElementTree as ET

import pandas as pd
import requests
from lithiumscope.core.paths import DATA_DIR, RESULTS_DIR
from lithiumscope.results.atomic_exports import write_table, write_legacy_evidence_json

ROOT=DATA_DIR/"raw/model_1/research/directed"
MAX_BYTES=30*1024*1024
MEIXNER="meixner_2020"
DOIS={
    MEIXNER:"10.1007/s00126-019-00915-2",
    "alvarez_amado_2022":"10.1016/j.gexplo.2022.107062",
    "fernandez_turiel_2020":"10.20350/DIGITALCSIC/10527",
    "macusani_2025":"10.60520/IEDA/113778",
}
DOWNLOADS={
    MEIXNER:{f"126_2019_915_MOESM{i}_ESM.{ext}":
       f"https://static-content.springer.com/esm/art%3A10.1007%2Fs00126-019-00915-2/MediaObjects/126_2019_915_MOESM{i}_ESM.{ext}"
       for i,ext in ((1,"docx"),(2,"xls"),(3,"xls"),(4,"xls"))},
}
ALLOWED={".xls",".xlsx",".docx",".csv",".tsv",".txt",".pdf"}
OLE=b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
LI_PATTERN=re.compile(r"(?<![a-z])(?:li|lithium|litio)(?![a-z])",re.I)
UNIT_PATTERN=re.compile(r"ppm|ppb|mg\s*/\s*kg|wt\s*%|µg\s*/\s*g|ug\s*/\s*g",re.I)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def check_file(path):
    suffix=path.suffix.lower()
    with path.open("rb") as stream:
        head=stream.read(512)
    if not head or head.lstrip().lower().startswith((b"<!doctype html",b"<html")):
        raise ValueError("Empty or HTML response, not an original table")
    if suffix==".xls" and not head.startswith(OLE):
        raise ValueError("Not an XLS compound document")
    if suffix in {".xlsx",".docx"}:
        if not is_zipfile(path):
            raise ValueError("Not an Office archive")
        with ZipFile(path) as z:
            expected="xl/workbook.xml" if suffix==".xlsx" else "word/document.xml"
            if expected not in z.namelist():
                raise ValueError("Unexpected Office document type")
    if suffix==".pdf" and not head.startswith(b"%PDF-"):
        raise ValueError("Not a PDF")
    if path.stat().st_size>MAX_BYTES:
        raise ValueError("Original exceeds bounded intake size")


def acquire(path,url):
    """Preserve existing originals; never overwrite or save an HTML error page."""
    if path.exists():
        check_file(path)
        return "existing"
    tmp=path.with_name(path.stem+".partial"+path.suffix)
    try:
        with requests.get(url,stream=True,timeout=(10,45),
                          headers={"User-Agent":"LithiumScope research data intake/1.0"}) as resp:
            resp.raise_for_status()
            size=0
            with tmp.open("wb") as stream:
                for block in resp.iter_content(65536):
                    size+=len(block)
                    if size>MAX_BYTES:
                        raise ValueError("Download exceeds bounded intake size")
                    stream.write(block)
        check_file(tmp)
        # Exclusive creation: no replacement of an original if another run raced.
        with path.open("xb") as stream:
            stream.write(tmp.read_bytes())
        return "downloaded"
    finally:
        tmp.unlink(missing_ok=True)


def cell_text(v):
    return "" if pd.isna(v) else str(v).strip()


def table_inventory(frame,sheet):
    # No assumed orientation, header, unit or grain; report candidate cells and
    # dimensions only. Full raw table is kept locally, not duplicated in evidence.
    matches=[]
    for i,row in enumerate(frame.itertuples(index=False,name=None)):
        for j,v in enumerate(row):
            s=cell_text(v)
            if LI_PATTERN.search(s):
                matches.append({"sheet":sheet,"row":i+1,"column":j+1,
                    "label":s[:150],"explicit_unit_in_cell":bool(UNIT_PATTERN.search(s))})
    return {"sheet":sheet,"rows":int(frame.shape[0]),"columns":int(frame.shape[1]),
            "nonempty_cells":int(frame.notna().sum().sum()),"li_mentions":len(matches)},matches


def inspect(path):
    suffix=path.suffix.lower()
    if suffix in {".xlsx",".xls"}:
        engine="xlrd" if suffix==".xls" else "openpyxl"
        with pd.ExcelFile(path,engine=engine) as wb:
            frames=[(name,wb.parse(name,header=None,dtype=object)) for name in wb.sheet_names]
    elif suffix in {".csv",".tsv",".txt"}:
        try:
            frame=pd.read_csv(path,sep=None,engine="python",header=None,dtype=str,encoding="utf-8-sig")
        except UnicodeDecodeError:
            frame=pd.read_csv(path,sep=None,engine="python",header=None,dtype=str,encoding="latin1")
        frames=[("text",frame)]
    elif suffix==".docx":
        ns={"w":"http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        with ZipFile(path) as z:
            root=ET.fromstring(z.read("word/document.xml"))
        frames=[]
        for i,t in enumerate(root.findall(".//w:tbl",ns),1):
            rows=[]
            for tr in t.findall("w:tr",ns):
                rows.append([" ".join(n.text or "" for n in tc.findall(".//w:t",ns))
                             for tc in tr.findall("w:tc",ns)])
            frames.append((f"table_{i}",pd.DataFrame(rows)))
        paragraphs=[" ".join(n.text or "" for n in p.findall(".//w:t",ns))
                    for p in root.findall("w:body/w:p",ns)]
        frames.append(("paragraphs",pd.DataFrame(paragraphs)))
    else:
        return [],[],"manual_pdf_review"
    summaries,mentions=[],[]
    for name,frame in frames:
        info,found=table_inventory(frame,name)
        summaries.append(info)
        mentions.extend(found)
    return summaries,mentions,"inspected_unmapped"


def run(*,download=False):
    out=RESULTS_DIR/"generalization/model_1_directed_source_intake"/(
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")+"_"+uuid4().hex[:8])
    out.mkdir(parents=True,exist_ok=False)
    inventory,sheets,mentions=[],[],[]
    for key,doi in DOIS.items():
        folder=ROOT/key
        folder.mkdir(parents=True,exist_ok=True)
        links=DOWNLOADS.get(key,{})
        outcomes={}
        for name,url in links.items():
            path=folder/name
            if download:
                try:
                    outcomes[name]=acquire(path,url)
                except (requests.RequestException,OSError,ValueError) as exc:
                    outcomes[name]="acquisition_failed: "+str(exc)[:300]
            elif not path.exists():
                outcomes[name]="not_downloaded"
        paths={p.name:p for p in folder.iterdir() if p.is_file() and
               p.suffix.lower() in ALLOWED and ".partial." not in p.name}
        for name in sorted(set(paths)|set(links)):
            path=folder/name
            record={"source":key,"doi":doi,"file":name,"url":links.get(name),
                    "acquisition":outcomes.get(name,"local_original_unverified"),
                    "license_review_required":True,"approved_for_training":False}
            inventory.append(record)
            if name not in paths:
                record["state"]="missing"
                continue
            try:
                check_file(path)
                before=sha(path.read_bytes())
                record.update(sha256=before,bytes=path.stat().st_size)
                tables,cells,state=inspect(path)
                if sha(path.read_bytes())!=before:
                    raise ValueError("Original changed during inspection")
                record["state"]=state
                sheets.extend({"source":key,"file":name,**v} for v in tables)
                mentions.extend({"source":key,"file":name,**v} for v in cells)
            except Exception as exc:
                record.update(state="inspection_failed",error=f"{type(exc).__name__}: {exc}"[:400])
        if not paths and not links:
            inventory.append({"source":key,"doi":doi,"state":"manual_original_required",
                              "approved_for_training":False,"license_review_required":True})
    tables={"source_inventory.csv":pd.DataFrame(inventory),
        "sheet_inventory.csv":pd.DataFrame(sheets,columns=["source","file","sheet","rows","columns","nonempty_cells","li_mentions"]),
        "li_cell_inventory.csv":pd.DataFrame(mentions,columns=["source","file","sheet","row","column","label","explicit_unit_in_cell"])}
    for name,table in tables.items():
        write_table(out/name,table)
    write_legacy_evidence_json(out/"manifest.json",{
        "state":"review_required","scope":"directed_original_intake_unmapped",
        "download_requested":download,"source_dois":DOIS,"files_inspected":sum(v.get("state")=="inspected_unmapped" for v in inventory),
        "new_rows_approved":0,"training":False,"harmonization":False,"production_schema_changed":False,
        "outputs":{name:sha((out/name).read_bytes()) for name in tables}})
    print(tables["source_inventory.csv"].to_string(index=False))
    print(tables["sheet_inventory.csv"].to_string(index=False))
    print(f"Estado: review_required | Evidencias: {out}",flush=True)
    return out


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download",action="store_true")
    run(download=parser.parse_args().download)
