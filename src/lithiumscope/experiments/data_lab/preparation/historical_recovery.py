"""Targeted review of original historical GEOROC files, not a dataset rebuild."""
from pathlib import Path
import numpy as np
import pandas as pd

from lithiumscope.datasets.adapters.georoc.schema import ingestion_layout, infer_georoc_column_map, MAJOR_OXIDES
from lithiumscope.datasets.adapters.georoc.transformation import harmonize_chunk
from lithiumscope.datasets.readers.georoc_raw_reader import detect_georoc_encoding
from lithiumscope.datasets.readers.csv_reader import validate_csv_structure
from .evidence import BASE_SHA, checked_file, read
from .recovery import independent_links

COLUMNS = ["base_position","oxide","proposed_value_wt_pct","original_value","original_column",
           "source_file","source_sha256","normalized_row_position","target_used",
           "basis_certified","approved_for_training"]


def review_originals(base, qc, cfg, project_root):
    """Only consult files named by the relevant exclusions and their merge lineage."""
    root = Path(project_root).resolve()
    settings = cfg["data_sources"]["georoc"]
    targets = qc.loc[qc.reason.eq("missing_or_unparseable") & qc.measured_oxide_count.ge(5)].copy()
    proposals, inventory = [], []
    if targets.empty:
        return pd.DataFrame(columns=COLUMNS), pd.DataFrame(columns=["file","state","reason"])
    try:
        merge = read(root / settings["audit_path"])
        if merge["combined_sha256"] != BASE_SHA:
            raise ValueError("Historical merge belongs to another source")
        harmonized = root / settings["harmonized_path"]
        checked_file(harmonized,merge["harmonized_sha256"])
        ingest = read(harmonized.with_suffix(".audit.json"))
        if ingest["output_sha256"] != merge["harmonized_sha256"]:
            raise ValueError("Historical ingestion belongs to another harmonized view")
        hashes = {}
        for name,digest in ingest["source_sha256"].items():
            basename = str(name).replace(chr(92),"/").rsplit("/",1)[-1]
            hashes.setdefault(basename,set()).add(digest)
    except (OSError,ValueError,TypeError,KeyError) as exc:
        return pd.DataFrame(columns=COLUMNS),pd.DataFrame([{"file":"historical_lineage",
            "state":"blocked","reason":str(exc)}])
    by_file = targets.groupby("reported_source_file",dropna=False)
    for name,subset in by_file:
        row={"file":str(name),"state":"reviewed","target_rows":len(subset),"proposed_measurements":0}
        try:
            if pd.isna(name) or Path(str(name)).name!=name or "/" in name or chr(92) in name:
                raise ValueError("No verified historical source filename")
            expected=hashes.get(name,set())
            if len(expected)!=1:
                raise ValueError("Historical partition digest unavailable or ambiguous")
            digest=next(iter(expected))
            original=checked_file((root / settings["input_glob"]).parent / name,digest)
            encoding=detect_georoc_encoding(original)
            header=validate_csv_structure(original,delimiter=",",encoding=encoding)
            mappings,_=ingestion_layout([original],lambda _:header,settings["iron_total_mode"])
            mapping=mappings[original]
            raw=pd.read_csv(original,usecols=list(mapping),encoding=encoding,dtype=str,low_memory=False)
            candidate,_=harmonize_chunk(raw,column_map=mapping,source_name="GEOROC",source_file=name,
                allowed_material_types=tuple(settings["allowed_material_types"]),minimum_predictors=0,
                iron_total_mode=settings["iron_total_mode"])
            if "source_sample" not in mapping.values():
                raise ValueError("Source has no explicit sample names")
            historic=base.loc[base.source_file.eq(name)]
            links=independent_links(historic,candidate)
            good=links.loc[links.unique_link].set_index("base_position")
            reverse={v:k for k,v in infer_georoc_column_map(header).items()}
            for item in subset.itertuples(index=False):
                idx=str(item.row_index)
                if idx not in good.index:
                    continue
                pos=int(candidate.loc[good.loc[idx,"candidate_position"],"source_row_position"])
                for oxide in str(item.missing_or_invalid_oxides).split("|"):
                    column=reverse.get(oxide)
                    if oxide not in MAJOR_OXIDES or oxide=="Fe2O3" or column is None or "WT%" not in column.upper().replace(" ",""):
                        continue
                    val=pd.to_numeric(raw.loc[pos,column],errors="coerce")
                    if pd.notna(val) and np.isfinite(val) and val>=0:
                        proposals.append({"base_position":idx,"oxide":oxide,"proposed_value_wt_pct":float(val),
                            "original_value":str(raw.loc[pos,column]),"original_column":column,"source_file":name,
                            "source_sha256":digest,"normalized_row_position":pos,"target_used":False,
                            "basis_certified":False,"approved_for_training":False})
                        row["proposed_measurements"]+=1
            checked_file(original,digest)
        except (OSError,ValueError,TypeError,KeyError,IndexError) as exc:
            row.update(state="blocked",reason=str(exc))
        inventory.append(row)
    return pd.DataFrame(proposals,columns=COLUMNS),pd.DataFrame(inventory)
