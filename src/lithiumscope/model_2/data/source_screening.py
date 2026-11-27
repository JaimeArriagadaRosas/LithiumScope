"""GEOROC citation screening and deterministic sample identity."""
from __future__ import annotations
import csv
import hashlib
import io
import re
import pandas as pd

REF_LINE = re.compile(r"^\[(\d+)\]\s*(.*)$", re.S)


def parse_bibliography(text: str) -> dict[str,str]:
    found={}
    for line in csv.reader(io.StringIO(text)):
        if line:
            m=REF_LINE.match(line[0])
            if m:found[m[1]]=m[2].strip()
    return found


def decimal_places(value) -> int:
    if pd.isna(value):return 0
    s=str(value).strip()
    return len(s.split(".")[1]) if re.fullmatch(r"[-+]?\d+\.\d+",s) else 0


def classify_record(bibliography_complete: bool, decimal_lat: int,
                    decimal_lon: int, repeated_site: int,
                    comment: str, ambiguous_id: bool=False) -> str:
    note=str(comment)
    if re.search(r"\b(?:DRILL CORE|DRILL HOLE|BOREHOLE|UNDERGROUND)\b",
                 note,re.I):
        return "EXCLUDE_SUBSURFACE_LABEL"
    if (not bibliography_complete or decimal_lat<=2 or decimal_lon<=2
            or repeated_site>=4 or ambiguous_id
            or re.search(r"\b(?:MINE|LITHIUM PROJECT|OPEN PIT)\b",note,re.I)):
        return "MANUAL_SOURCE_REVIEW"
    return "PROVISIONAL_SITE_REVIEW"


def stable_sample_id(ref, name, lat, lon) -> str:
    key = f"GEOROC|{ref}|{name}|{float(lat):.6f}|{float(lon):.6f}"
    return "GEOROC_" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
