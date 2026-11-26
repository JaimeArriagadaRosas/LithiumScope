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
from lithiumscope.datasets.management.local_sources import local_training_source
from lithiumscope.datasets.adapters.georoc_ingestion_schema import infer_georoc_column_map
from lithiumscope.datasets.readers.georoc_raw_reader import detect_georoc_encoding
from lithiumscope.datasets.validation.analytical import parse_analytical_value
from lithiumscope.experiments.data_lab.candidate_screening import CHEM, sample, num
from lithiumscope.experiments.model_1.balanced.academic import cohort
from lithiumscope.experiments.model_1.expansion.raw_recovery import EXPANSION, EXPANSION_SHA
from lithiumscope.experiments.model_1.expansion.sensitivity_data import load_evidence, read
from lithiumscope.experiments.model_1.augmentation.source_intake import ROOT, MEIXNER
from lithiumscope.results.atomic_exports import write_table, write_legacy_evidence_json

ORIGINALS={
    "126_2019_915_MOESM1_ESM.docx":"2070c8e531c40ef2816621e3d786e626c43a4c622d69344df50d12ac4a00272e",
    "126_2019_915_MOESM2_ESM.xls":"ffb7171170cfb383bc47ccc4f86f6aa7f1197f08c9a4bcdfed9e65a14efffbe1",
    "126_2019_915_MOESM3_ESM.xls":"7e172ef76fb1387aa6a264ed9bce62b9a4f5e8c25a5f725c192e3b6bb195767d",
    "126_2019_915_MOESM4_ESM.xls":"155c7c5a44c090a0a99726c49cc247e60df551ad907296507a861e31ae557286",
}
SECTIONS=((13,34,"monogenetic_lavas","Risse et al. 2013"),
    (38,41,"ignimbrite","Lindsay et al. 2001"),
    (43,48,"ignimbrite","Grocke et al. 2017"),
    (50,50,"ignimbrite","Meixner et al. 2020"),
    (54,76,"andesite","Rosner et al. 2003"),
    (80,94,"sedimentary_bolivia","Egenhoff and Lucassen 2003"),
    (98,100,"sedimentary_argentina","Meixner et al. 2020"),
    (104,116,"metamorphic_basement","Kasemann et al. 2000"),
    (120,121,"granite","Meixner et al. 2020"))


def number(v):
    n=pd.to_numeric(v,errors="coerce")
    return float(n) if pd.notna(n) and np.isfinite(n) else np.nan


def parse_a1(table):
    if table.shape[0]<121 or table.shape[1]<14:
        raise ValueError("Unexpected A1 layout")
    for col,expected in {0:"sample identifier",1:"latitude",2:"longitude",6:"Li [µg/g]",11:"SiO2 [wt%]"}.items():
        if str(table.iat[8,col]).strip()!=expected:
            raise ValueError("A1 header differs")
    rows=[]
    for start,end,rock,ref in SECTIONS:
        for excel_row in range(start,end+1):
            r=table.iloc[excel_row-1]
            if not isinstance(r.iloc[0],str) or not np.isfinite(number(r.iloc[4])):
                continue
            name=r.iloc[0].strip()
            rows.append({"excel_row":excel_row,"source_sample":name,"rock_section":rock,
                "section_reference_hint":ref,"repeat":"*" in name,
                "Latitude":number(r.iloc[1]),"Longitude":number(r.iloc[2]),
                "Li_ppm":number(r.iloc[6]),"SiO2":number(r.iloc[11]),
                "approved_for_training":False})
    result=pd.DataFrame(rows)
    if len(result)!=89 or int(result.repeat.sum())!=7:
        raise ValueError("A1 row inventory differs")
    valid=result.loc[~result.repeat&result.Li_ppm.notna()]
    if len(valid)!=80 or not valid.Li_ppm.gt(0).all() or valid.source_sample.duplicated().any():
        raise ValueError("A1 primary samples differ")
    return result


def raw_view(chunk,mapping):
    """Do not call harmonize_chunk: it drops rows without Li before identity review."""
    frame=chunk.rename(columns=mapping).copy()
    for c in (*CHEM,"Li_icpms"):
        if c in frame:
            frame[c]=frame[c].map(lambda v:parse_analytical_value(v,decimal_comma=False)[0])
    for axis in ("Latitude","Longitude"):
        if axis not in frame and axis+"_min" in frame:
            low=pd.to_numeric(frame[axis+"_min"],errors="coerce")
            high=pd.to_numeric(frame.get(axis+"_max",low),errors="coerce").fillna(low)
            frame[axis]=(low+high)/2
    return frame


def match_names(query,frame,source,filename):
    """Report all normalized-name matches, never select by Li."""
    keys=sample(frame)
    coords=num(frame,["Latitude","Longitude","SiO2"])
    chemistry=num(frame,CHEM)
    count=(chemistry.notna()&chemistry.ge(0)).sum(axis=1)
    records=[]
    for _,q in query.iterrows():
        name=sample(pd.DataFrame([q])).iloc[0]
        if not name:
            continue
        for idx in keys.index[keys.eq(name)]:
            row=frame.loc[idx]
            a=coords.loc[idx]
            complete=np.isfinite([q.Latitude,q.Longitude,a.Latitude,a.Longitude]).all()
            site=bool(complete and abs(q.Latitude-a.Latitude)<=.001 and abs(q.Longitude-a.Longitude)<=.001)
            silica=bool(np.isfinite([q.SiO2,a.SiO2]).all() and abs(q.SiO2-a.SiO2)<=.5)
            qc=number(row.get("SUM (no water)",np.nan))
            # For raw views, strict full oxide sum is only a diagnostic.
            if not np.isfinite(qc):
                ox=chemistry.loc[idx,list(CHEM[:10])]
                qc=float(ox.sum()) if ox.notna().all() and ox.ge(0).all() else np.nan
            records.append({"excel_row":int(q.excel_row),"source_sample":q.source_sample,
                "candidate_source":source,"candidate_file":filename,"candidate_position":str(idx),
                "candidate_sample":str(row.get("source_sample",row.get("Sample",""))),
                "candidate_publication":str(row.get("source_publication",row.get("Geochemistry Reference",""))),
                "candidate_material":str(row.get("Sample_type","")),"candidate_rock":str(row.get("Rock_type","")),
                "site_match":site,"silica_match":silica,"chemical_predictors":int(count.loc[idx]),
                "oxide_sum":qc,"qc_diagnostic":bool(np.isfinite(qc) and 94<=qc<=102),
                "candidate_has_li":bool(np.isfinite(number(row.get("Li_icpms",np.nan)))),
                "identity_review_candidate":bool(site and silica),
                "approved_for_training":False})
    return records


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
