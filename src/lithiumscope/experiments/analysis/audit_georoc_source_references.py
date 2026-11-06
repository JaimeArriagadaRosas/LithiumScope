"""Conservative citation and site audit for GEOROC-to-Sentinel-2 labels."""
from __future__ import annotations
import csv
import hashlib
import io
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
from lithiumscope.core.paths import PROJECT_ROOT, RESULTS_DIR
from lithiumscope.experiments.analysis.georoc_qc_protocols import protocol_masks
from lithiumscope.experiments.analysis.pilot_georoc_m2_labels import _stable_sample_id

OUT = RESULTS_DIR/"scientific_simulations"/"vnext_20261106"/"georoc_source_validation"
REF_URL = "https://georoc.mpch-mainz.gwdg.de/georoc/reference_list.csv"
REF_ID = re.compile(r"\[(\d+)\]")
REF_LINE = re.compile(r"^\[(\d+)\]\s*(.*)$",re.S)
DOI = re.compile(r"doi:\s*(10\.\d+/\S+)",re.I)


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


def audit(root: Path=PROJECT_ROOT, output: Path=OUT) -> dict:
    ref_file=output/"georoc_reference_list_snapshot.csv"
    if not ref_file.exists():
        raise FileNotFoundError("Official references not cached; bibliography download required.")
    blob=ref_file.read_bytes()
    refs=parse_bibliography(blob.decode("cp1252",errors="replace"))
    if len(refs)<20000:raise ValueError("Incomplete official bibliography")
    source=root/"results/scientific_simulations/vnext_20261106/stage2_m1/feot_combined_experiment_only.csv"
    data=pd.read_csv(source,low_memory=False)
    good=protocol_masks(data)["B_reconstructed"]&data.source_dataset.eq("GEOROC")
    geo=data.loc[good].copy()
    raw=pd.read_csv(root/"data/raw/model_1/georoc/GEOROC_Andean_Arc_LithiumScope.csv",dtype=str)
    raw=raw.rename(columns={"SAMPLE NAME":"source_sample"})
    keys=["CITATIONS","source_sample"]
    raw_counts=raw.groupby(keys).size().rename("raw_matches").reset_index()
    cols=keys+["LATITUDE MIN","LONGITUDE MIN","LOCATION COMMENT","ROCK CLASS"]
    raw_meta=raw[cols].drop_duplicates(keys,keep="first").rename(columns={"LOCATION COMMENT":"raw_comment","ROCK CLASS":"raw_rock_class"})
    geo=geo.merge(raw_meta,on=keys,how="left",validate="many_to_one")
    geo=geo.merge(raw_counts,on=keys,how="left",validate="many_to_one")
    if geo["LATITUDE MIN"].isna().any():raise ValueError("Raw citation/sample join missing")
    geo["sample_id"]=[
        _stable_sample_id(str(ref),str(name),float(lat),float(lon))
        for ref,name,lat,lon in zip(geo.CITATIONS,geo.source_sample,
                                    geo.Latitude,geo.Longitude,strict=True)
    ]
    assert not geo.sample_id.duplicated().any()
    coord=geo.Latitude.round(5).astype(str)+"|"+geo.Longitude.round(5).astype(str)
    geo["site_count"]=coord.map(coord.value_counts())
    geo["decimal_lat"]=geo["LATITUDE MIN"].map(decimal_places)
    geo["decimal_lon"]=geo["LONGITUDE MIN"].map(decimal_places)
    geo["citation_ids"]=geo.CITATIONS.astype(str).map(
        lambda t:REF_ID.findall(t))
    geo["references_complete"]=geo.citation_ids.map(
        lambda ids:len(ids)>0 and all(k in refs for k in ids))
    geo["source_dois"]=geo.citation_ids.map(
        lambda ids:";".join(
            m.group(1).rstrip(".,;") for k in ids if k in refs
            for m in [DOI.search(refs[k])] if m))
    geo["screening_tier"]=[
        classify_record(bool(ref_ok),int(latdigits),int(londigits),int(cnt),
                        comment,bool(nraw>1))
        for ref_ok,latdigits,londigits,cnt,comment,nraw in zip(
            geo.references_complete,geo.decimal_lat,geo.decimal_lon,
            geo.site_count,geo["raw_comment"].fillna(""),
            geo.raw_matches.fillna(0),strict=True)
    ]
    geo["geologic_label_certified"]=False
    keep=["sample_id","CITATIONS","source_sample","Latitude","Longitude",
          "Li_icpms","LOCATION","raw_comment","raw_rock_class",
          "source_dois","site_count","raw_matches",
          "decimal_lat","decimal_lon","references_complete",
          "screening_tier","geologic_label_certified"]
    output.mkdir(parents=True,exist_ok=True)
    geo[keep].to_csv(output/"georoc_B_screened_sites.csv",index=False)
    all_ids=sorted({k for ids in geo.citation_ids for k in ids},key=int)
    pd.DataFrame([{
        "ref_id":k,"matched":k in refs,"citation_text":refs.get(k,""),
        "doi":DOI.search(refs[k]).group(1).rstrip(".,;")
              if k in refs and DOI.search(refs[k]) else "",
    } for k in all_ids]).to_csv(output/"GEOROC_B_bibliography.csv",index=False)
    pilot=pd.read_csv(root/"results/scientific_simulations/vnext_20261106/m2_georoc_labelled_pilot/selected_candidates.csv")
    review=pilot[["sample_id","source_citation"]].merge(
        geo[keep],on="sample_id",how="left",validate="one_to_one")
    if review.screening_tier.isna().any():raise ValueError("Pilot samples absent from B")
    review.to_csv(output/"six_pilot_site_reviews.csv",index=False)
    summary={
        "bibliography_url":REF_URL,
        "bibliography_sha256":hashlib.sha256(blob).hexdigest(),
        "catalogue_reference_ids":len(refs),
        "B_georoc_records":len(geo),
        "B_unique_citation_ids":len(all_ids),
        "B_citation_ids_resolved":sum(x in refs for x in all_ids),
        "B_samples_with_complete_bibliography":int(geo.references_complete.sum()),
        "B_coarse_nominal_coordinate_decimals":int(
            (geo[["decimal_lat","decimal_lon"]].min(axis=1)<=2).sum()),
        "B_repeated_site_at_least_four":int((geo.site_count>=4).sum()),
        "B_ambiguous_raw_identity":int((geo.raw_matches>1).sum()),
        "B_screening_tiers":geo.screening_tier.value_counts().to_dict(),
        "pilot_screening_tiers":review.screening_tier.value_counts().to_dict(),
        "pilot_rows":review[["source_sample","CITATIONS","screening_tier",
                            "references_complete","site_count","decimal_lat",
                            "decimal_lon","raw_comment"]].to_dict("records"),
        "approved_for_training":0,
        "scientific_limit":"Bibliography match, coordinate decimals and a saved patch never prove field GPS location or geologic representativeness.",
        "no_production_changes":True,
    }
    (output/"GEOROC_source_validation_summary.json").write_text(
        json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf8")
    print("SOURCE_AUDIT",json.dumps(summary,ensure_ascii=False),flush=True)
    return summary


if __name__=="__main__":
    audit()
