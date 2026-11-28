"""Target-independent identity audit and original measurement recovery proposals."""
from tempfile import TemporaryDirectory
from pathlib import Path
import re

import numpy as np
import pandas as pd

from lithiumscope.core.hashing import file_sha256
from lithiumscope.datasets.adapters.georoc.research_view import normalize_research_export
from lithiumscope.datasets.adapters.georoc.schema import infer_georoc_column_map, MAJOR_OXIDES
from ..screening.candidates import CHEM, num, text
from .evidence import read, checked_file
from .reconciliation import view

ANDEAN = "georoc_andean_202609"
SHARED = [c for c in CHEM if c != "Fe2O3"]


def independent_links(base, candidate):
    """Match only explicit sample names, nominal location and measured chemistry."""
    a = text(base, "source_sample").where(text(base,"source_sample").ne(""), text(base,"Sample"))
    b = text(candidate, "source_sample")
    ac, bc = num(base, ["Latitude","Longitude"]), num(candidate, ["Latitude","Longitude"])
    ax, bx = num(base, SHARED), num(candidate, SHARED)
    groups = b.groupby(b).groups
    edges = []
    for idx,name in a.items():
        if not name or name not in groups or not ac.loc[idx].notna().all():
            continue
        ids = list(groups[name])
        loc = bc.loc[ids].to_numpy(float)
        same = np.isclose(loc, ac.loc[idx].to_numpy(float), atol=1e-5, rtol=0).all(axis=1)
        left,right = ax.loc[idx].to_numpy(float), bx.loc[ids].to_numpy(float)
        shared = np.isfinite(left) & np.isfinite(right)
        agreement = np.abs(left-right) <= np.maximum(.001,.02*np.maximum(np.abs(left),np.abs(right)))
        keep = same & (shared.sum(axis=1)>=5) & (agreement|~shared).all(axis=1)
        for i in np.flatnonzero(keep):
            edges.append({"base_position":str(idx),"candidate_position":ids[i],
                          "shared_chemistry":int(shared[i].sum()),"target_used":False})
    result = pd.DataFrame(edges, columns=["base_position","candidate_position","shared_chemistry","target_used"])
    result["unique_link"] = ~result.base_position.duplicated(keep=False) & ~result.candidate_position.duplicated(keep=False)
    return result


def _bibliography(record):
    index = {}
    for row in record.get("footer", []):
        if not str(row["section"]).strip().lower().startswith(("references","bibliography","citations")):
            continue
        for cell in row["cells"]:
            match = re.match(r"^\s*\[(\d+)\]\s*(.+)",cell)
            if match:
                index.setdefault(match[1],set()).add(match[2].strip())
    return index


def recover(base, qc_review, profile):
    candidate = view(profile)
    # Match all historical GEOROC rows, including exclusions, so a second
    # historical sample competing for a source row is not silently ignored.
    historical = base.loc[text(base,"source_dataset").eq("georoc")]
    links = independent_links(historical, candidate)
    qc = qc_review.loc[qc_review.reason.eq("missing_or_unparseable")].copy()
    qc["base_position"] = qc.row_index.astype(str)
    qc = qc.set_index("base_position")
    structures = read(profile["path"] / "tables/raw_structure_audit.json")
    good = links.loc[links.unique_link].copy()
    good["file"] = good.candidate_position.map(candidate.source_file)
    publications, proposals = [], []
    for file, rows in good.groupby("file"):
        if Path(file).name != file or file not in structures:
            raise ValueError("Invalid recovery partition")
        digest = profile["manifest"]["source_sha256"][file]
        original = checked_file(profile["original_folder"] / file,digest)
        info = structures[file]
        if info["original_sha256"] != digest:
            raise ValueError("Raw structure identity differs")
        refs = _bibliography(info)
        with TemporaryDirectory() as temp:
            normalized = Path(temp) / file
            audit = normalize_research_export(original, normalized)
            if audit["data_records"] != info["data_records"]:
                raise ValueError("Normalized source positions differ")
            raw = pd.read_csv(normalized, dtype=str, low_memory=False)
        mapping = infer_georoc_column_map(raw.columns)
        reversed_map = {v:k for k,v in mapping.items()}
        for link in rows.itertuples(index=False):
            c = candidate.loc[link.candidate_position]
            position = int(c.source_row_position)
            if not 0 <= position < len(raw) or float(c.source_row_position) != position:
                raise ValueError("Invalid normalized record position")
            original_row = raw.iloc[position]
            name_column = reversed_map.get("source_sample")
            if name_column is None or str(original_row[name_column]).strip().casefold() != str(c.source_sample).strip().casefold():
                continue
            pub = str(c.get("source_publication",""))
            ids = re.findall(r"\[(\d+)\]",pub)
            if ids and all(len(refs.get(i,set()))==1 for i in ids):
                publications.append({"base_position":link.base_position,"source_file":file,
                    "source_sha256":digest,"normalized_row_position":position,
                    "publication_ids":pub,"references":" | ".join(next(iter(refs[i])) for i in ids),
                    "target_used":False,"identity_review_required":True})
            if link.base_position not in qc.index:
                continue
            item = qc.loc[link.base_position]
            for oxide in str(item.missing_or_invalid_oxides).split("|"):
                column = reversed_map.get(oxide)
                # Only explicit wt% numeric measurements, not censor substitutions,
                # iron conversion or inferred totals. Human review of units/basis remains.
                if oxide not in MAJOR_OXIDES or oxide == "Fe2O3" or column is None or "WT%" not in column.upper().replace(" ",""):
                    continue
                val = pd.to_numeric(original_row[column],errors="coerce")
                if pd.notna(val) and np.isfinite(val) and val >= 0:
                    proposals.append({"base_position":link.base_position,"oxide":oxide,
                        "proposed_value_wt_pct":float(val),"original_value":str(original_row[column]),
                        "original_column":column,"source_file":file,"source_sha256":digest,
                        "normalized_row_position":position,"target_used":False,
                        "basis_certified":False,"approved_for_training":False})
        checked_file(original,digest)
    pub_cols = ["base_position","source_file","source_sha256","normalized_row_position",
                "publication_ids","references","target_used","identity_review_required"]
    proposal_cols = ["base_position","oxide","proposed_value_wt_pct","original_value",
                     "original_column","source_file","source_sha256","normalized_row_position",
                     "target_used","basis_certified","approved_for_training"]
    return links,pd.DataFrame(publications,columns=pub_cols),pd.DataFrame(proposals,columns=proposal_cols)
