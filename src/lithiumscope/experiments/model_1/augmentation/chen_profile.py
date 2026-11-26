"""Chen Supplementary Data 3: bounded, review-only EDA of the original workbook."""
from datetime import datetime, timezone
from uuid import uuid4
import re
import numpy as np
import pandas as pd

from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.paths import RESULTS_DIR
from lithiumscope.datasets.adapters.georoc_ingestion_schema import infer_georoc_column_map
from lithiumscope.experiments.data_lab.candidate_screening import CHEM, num, group, sample
from lithiumscope.experiments.model_1.augmentation.source_intake import ROOT
from lithiumscope.experiments.model_1.augmentation.meixner_audit import raw_view
from lithiumscope.experiments.model_1.expansion.raw_recovery import measurement_map, numeric_ppm
from lithiumscope.results.atomic_exports import write_table, write_legacy_evidence_json

SOURCE="chen_global_arcs_2020"
NAME="41467_2020_19106_MOESM6_ESM.xlsx"
SHA="a5cfd0c5453ea26ed9f2e436e2de87678fb0a34e91b148d841dfda5f5db0e69d"
SHEET="Table 3 - global arc WR dataset"
REF=re.compile(r"^\s*\[\d+\]\s+[A-Za-z]")
BANDS=[0,5,10,20,40,60,80,100,126.12,696,np.inf]


def profile(table):
    if table.shape[0]<3 or table.shape[1]<18 or str(table.iat[1,17]).strip().upper()!="LI(PPM)":
        raise ValueError("Chen header layout differs")
    headers=[str(v).strip() if pd.notna(v) else "" for v in table.iloc[1]]
    # Blank/duplicate labels receive stable positional identifiers for profiling.
    counts=pd.Series(headers).value_counts()
    names=[h if h and counts[h]==1 else f"__col_{j+1}__{h}" for j,h in enumerate(headers)]
    raw=table.iloc[2:].copy().set_axis(names,axis=1)
    raw.index=raw.index+1 # original one-based Excel row
    mapping=infer_georoc_column_map(names)
    if mapping.get(names[17])!="Li_icpms":
        raise ValueError("Explicit ppm target not mapped")
    # Numeric labels only: censored targets are not measured exact values.
    target=pd.to_numeric(raw[names[17]],errors="coerce").astype(float)
    observed=np.isfinite(target)&target.gt(0)
    canon=raw_view(raw[list(mapping)],mapping)
    # FeOT is total iron reported as FeO. Preserve the original representation,
    # compute Fe2O3T equivalent for canonical comparison; never add both.
    if "FEOT(WT%)" in raw:
        feot=pd.to_numeric(raw["FEOT(WT%)"],errors="coerce").astype(float)
        canon["FeOT_total_as_FeO"]=feot.where(np.isfinite(feot)&feot.ge(0))
        if "Fe2O3" in canon:
            raise ValueError("Ambiguous iron representation")
        canon["Fe2O3"]=canon.FeOT_total_as_FeO*1.111348
        canon["Fe_total_provenance"]="FEOT_CONVERTED_TO_FE2O3T"
    chemistry=num(canon,CHEM)
    coverage=chemistry.notna().sum(axis=1)
    ox=chemistry[list(CHEM[:10])].sum(axis=1,min_count=10)
    coordinates=num(canon,["Latitude","Longitude"])
    coord_ok=coordinates.Latitude.between(-90,90)&coordinates.Longitude.between(-180,180)
    refs=raw.iloc[:,0].astype(str).str.match(REF)&~observed&coverage.eq(0)
    review=canon.loc[observed].copy()
    review["Li_icpms"]=target.loc[observed]
    review["excel_row"]=review.index
    review["source_dataset"]=SOURCE
    review["chemical_predictors"]=coverage.loc[observed]
    review["oxide_sum"]=ox.loc[observed]
    native=chemistry[list(CHEM[:10])].copy()
    if "FeOT_total_as_FeO" in canon:
        native["Fe2O3"]=canon.FeOT_total_as_FeO
    review["oxide_sum_original_basis"]=native.sum(axis=1,min_count=10).loc[observed]
    review["original_basis_qc"]=review.oxide_sum_original_basis.between(94,102)
    review["chen_original_basis_98_101_5"]=review.oxide_sum_original_basis.between(98,101.5)
    for element,(column,unit) in measurement_map(names).items():
        review["georoc_"+element+"_ppm"]=numeric_ppm(raw.loc[observed,column],unit)
    if "LOCATION COMMENT" in raw:
        review["source_location_comment"]=raw.loc[observed,"LOCATION COMMENT"]
    review["qc_diagnostic"]=review.oxide_sum.between(94,102)
    review["coordinates_valid"]=coord_ok.loc[observed]
    review["spatial_group"]=group(review)
    review["name_available"]=sample(review).ne("")
    review["academic_range"]=review.Li_icpms.between(4.2439,126.12)
    review["academic_tail"]=review.Li_icpms.gt(40)&review.academic_range
    review["basic_qc"]=review.qc_diagnostic&review.original_basis_qc&review.coordinates_valid&review.chemical_predictors.ge(8)
    review["approved_for_training"]=False
    # Original strings are preserved by row reference and the immutable workbook.
    inventory=[]
    for j,(header,name) in enumerate(zip(headers,names,strict=True)):
        v=raw[name]
        candidate=v.loc[observed]
        inventory.append({"excel_column":j+1,"header":header,"mapped_field":("FeOT_total_as_FeO" if name=="FEOT(WT%)" else mapping.get(name,"")),
            "nonempty_rows":int(v.notna().sum()),"nonempty_positive_li_rows":int(candidate.notna().sum()),
            "numeric_positive_li_rows":int(pd.to_numeric(candidate,errors="coerce").notna().sum()),
            "examples":" | ".join(str(x)[:100] for x in candidate.dropna().drop_duplicates().head(3))})
    counts_by_band=review.assign(band=pd.cut(review.Li_icpms,BANDS)).groupby("band",observed=False).agg(
        positive_li_rows=("Li_icpms","size"),basic_qc_rows=("basic_qc","sum"),
        coordinates_valid=("coordinates_valid","sum"),name_available=("name_available","sum")).reset_index()
    summary={"sheet_rows":len(table),"post_header_rows":len(raw),"reference_like_rows_without_chemistry":int(refs.sum()),
        "positive_numeric_li_rows":int(observed.sum()),"basic_qc_rows":int(review.basic_qc.sum()),
        "academic_range_rows":int(review.academic_range.sum()),
        "academic_tail_rows":int(review.academic_tail.sum()),
        "academic_tail_basic_qc_rows":int((review.academic_tail&review.basic_qc).sum()),
        "academic_basic_qc_rows":int((review.academic_range&review.basic_qc).sum()),
        "rows_with_name":int(review.name_available.sum()),"rows_with_coordinates":int(review.coordinates_valid.sum()),
        "li_min":float(review.Li_icpms.min()) if len(review) else None,
        "li_median":float(review.Li_icpms.median()) if len(review) else None,
        "li_max":float(review.Li_icpms.max()) if len(review) else None,
        "mapped_chemical_fields":[v for v in CHEM if v in canon],
        "original_basis_qc_rows":int(review.original_basis_qc.sum()),
        "chen_original_basis_qc_rows":int(review.chen_original_basis_98_101_5.sum()),
        "mapped_fields":mapping,"unique_sample_count_verified":False}
    return review,pd.DataFrame(inventory),counts_by_band,summary


def run():
    path=ROOT/SOURCE/NAME
    if file_sha256(path)!=SHA:
        raise ValueError("Chen original differs")
    table=pd.read_excel(path,sheet_name=SHEET,header=None,engine="openpyxl",dtype=object)
    if table.shape!=(50145,66):
        raise ValueError("Chen inventory dimensions differ")
    rows,inventory,bands,summary=profile(table)
    out=RESULTS_DIR/"generalization/model_1_chen_profile"/(
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")+"_"+uuid4().hex[:8])
    out.mkdir(parents=True,exist_ok=False)
    tables={"positive_li_review.csv":rows.reset_index(drop=True),"column_inventory.csv":inventory,
            "band_counts.csv":bands}
    for name,frame in tables.items():
        write_table(out/name,frame)
    if file_sha256(path)!=SHA:
        raise ValueError("Chen original changed")
    write_legacy_evidence_json(out/"manifest.json",{"state":"review_required","scope":"chen_original_profile_not_admission",
        "original_sha256":SHA,"summary":summary,"new_rows_approved":0,"training":False,
        "duplicate_audit_complete":False,"metadata_certified":False,"production_schema_changed":False,
        "outputs":{name:file_sha256(out/name) for name in tables}})
    print(pd.Series({k:v for k,v in summary.items() if k not in ("mapped_fields","mapped_chemical_fields")}).to_string())
    print(inventory[["excel_column","header","mapped_field","nonempty_positive_li_rows"]].to_string(index=False))
    print(bands.to_string(index=False))
    print(f"Estado: review_required | Evidencias: {out}",flush=True)
    return out


if __name__=="__main__":
    run()
