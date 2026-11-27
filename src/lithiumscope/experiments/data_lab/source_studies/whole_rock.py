"""Directed whole-rock source interpretation and review-only matching."""
from __future__ import annotations
import numpy as np
import pandas as pd
from lithiumscope.datasets.validation.chemistry.analytical import parse_analytical_value
from lithiumscope.experiments.data_lab.screening.candidates import CHEM, sample, num, group

PUNA="fernandez_turiel_2020"
MAC="macusani_2025"
FILES={
    PUNA:("Shoshonites_Puna-geochemistry-1-whole_rock.xlsx","4338150d9b3822a4e1cf7ef1f91283e98b4ec44fac3caf5ade9da8df0b831e45"),
    MAC:("3778-1_EC_BulkSample_Analyses_TorroetalMACUSANI.xlsx","ee8a26efb260015035930bc5199e4f94fd078b8dabccfd839aaeec9305ba1666"),
}
META=("Shoshonites_Puna-geochemistry-0_metadata.xlsx","4aa129fa5930313d892ceee79f98f59773a3ede672abcd8d45bf2c2d9f5afa56")
PUBLICATIONS={PUNA:"10.20350/DIGITALCSIC/10527",MAC:"10.1093/petrology/egaf041"}
OXIDES=tuple(CHEM[:10])
ALIASES={c:c for c in OXIDES}
ALIASES.update({"Fe2O3T":"Fe2O3","Li":"Li_icpms"})
ALIASES.update({c.split("_")[0]:c for c in CHEM[10:]})


def clean(value):
    v,kind,limit=parse_analytical_value(value,decimal_comma=False)
    n=pd.to_numeric(v,errors="coerce")
    return (float(n) if pd.notna(n) and np.isfinite(n) and n>=0 else np.nan),kind,limit


def norm_unit(value):
    return str(value).strip().lower().replace("μ","u").replace("µ","u").replace(" ","")


def parse_tables(samples,data,key):
    """Require explicit units, join exact sample IDs, keep original row numbers."""
    if key not in FILES:
        raise ValueError("Unknown source")
    puna=key==PUNA
    sh,first=(6,8) if puna else (7,6) # zero-based first sample/data row
    h,m,u=(3,4,5) if puna else (1,2,3)
    start=7 if puna else 8
    if samples.iat[4,0]!="SAMPLE NAME" or data.iat[6 if puna else 4,0]!="SAMPLE NAME":
        raise ValueError("Workbook sample headers differ")
    fields={}
    inventory=[]
    for j in range(start,data.shape[1]):
        label=str(data.iat[h,j]).strip()
        unit=norm_unit(data.iat[u,j])
        method=str(data.iat[m,j])
        canonical=ALIASES.get(label)
        if canonical:
            allowed={"wt%"} if canonical in OXIDES else {"mg/kg","ug/g","ppm"}
            if unit not in allowed or canonical in fields:
                raise ValueError("Unexpected unit or duplicate parameter: "+label+" "+unit)
            fields[canonical]=j
        inventory.append({"source_dataset":key,"parameter":label,"unit":unit,"method_code":method,
                          "canonical_field":canonical or ""})
    if set(fields)!=set(CHEM)|{"Li_icpms"}:
        raise ValueError("Canonical parameter inventory differs")
    by_name={}
    for i in range(sh,len(samples)):
        row=samples.iloc[i]
        if pd.isna(row.iloc[0]):
            continue
        name=str(row.iloc[0]).strip()
        if name in by_name:
            raise ValueError("Duplicate sample ID")
        by_name[name]=(i,row)
    rows,censoring=[],[]
    used=set()
    for i in range(first,len(data)):
        row=data.iloc[i]
        if pd.isna(row.iloc[0]):
            continue
        name=str(row.iloc[0]).strip()
        if name not in by_name or name in used:
            raise ValueError("Missing or repeated analysis ID: "+name)
        used.add(name)
        si,s=by_name[name]
        if puna and (pd.isna(s.iloc[1]) or str(s.iloc[1])!=str(row.iloc[1])):
            raise ValueError("IGSN mismatch: "+name)
        material=str(row.iloc[2]).strip().lower()
        record={"source_dataset":key,"source_sample":name,
                "source_publication":PUBLICATIONS[key],"sample_excel_row":si+1,"analysis_excel_row":i+1,
                "source_unique_id":str(s.iloc[1]) if pd.notna(s.iloc[1]) else "",
                "Latitude":pd.to_numeric(s.iloc[2],errors="coerce"),
                "Longitude":pd.to_numeric(s.iloc[3],errors="coerce"),
                "Rock_type":str(s.iloc[6]),"Sample_type":"whole rock" if material in {"whole rock","whole rock analyses"} else material,
                "source_li_unit":"ppm","source_li_method":"HR-ICP-MS" if puna else "review_method_code_3",
                "source_coordinate_crs":"unconfirmed",
                "source_li_raw":str(row.iloc[fields["Li_icpms"]]),
                "approved_for_training":False}
        for canonical,j in fields.items():
            value,kind,limit=clean(row.iloc[j])
            record[canonical]=value
            if canonical=="Li_icpms":
                record["source_li_censoring"]=kind or ""
            if kind:
                censoring.append({"source_dataset":key,"source_sample":name,"parameter":canonical,
                    "raw":str(row.iloc[j]),"kind":kind,"limit":limit,"value":value})
        # Keep additional ppm measurements in a separate review-only namespace.
        # Method codes are recorded in measurement_inventory; canonical names
        # are comparison aliases, not certification of analytical equivalence.
        for item,j in zip(inventory,range(start,data.shape[1]),strict=True):
            label=item["parameter"]
            if not item["canonical_field"] and item["unit"] in {"mg/kg","ug/g","ppm"} and label.isalpha() and 1<=len(label)<=2:
                record["extra_"+label+"_ppm"]=clean(row.iloc[j])[0]
        rows.append(record)
    if used!=set(by_name):
        raise ValueError("Unjoined sample IDs")
    result=pd.DataFrame(rows)
    if len(result)!=(6 if puna else 29):
        raise ValueError("Unexpected number of joined samples")
    return result,pd.DataFrame(inventory),censoring


def duplicate_edges(query,base,origin):
    """Name/IGSN or spatial chemical agreement, without inspecting either Li value."""
    names=sample(base)
    qnames=sample(query)
    b=num(base,("Latitude","Longitude",*CHEM))
    q=num(query,("Latitude","Longitude",*CHEM))
    ids=base.get("source_unique_id",pd.Series("",index=base.index)).fillna("").astype(str).str.strip().str.casefold()
    ids=ids.replace({"nan":"","none":"","<na>":""})
    records=[]
    for idx,row in query.iterrows():
        a=q.loc[idx]
        site=(b.Latitude.sub(a.Latitude).abs().le(.001)&b.Longitude.sub(a.Longitude).abs().le(.001))
        name=names.eq(qnames.loc[idx])&bool(qnames.loc[idx])
        uid=str(row.source_unique_id).strip().casefold()
        exact_id=ids.eq(uid)&bool(uid)
        for j in b.index[site|name|exact_id]:
            # Iron excluded because total versus component representation can differ.
            cols=[c for c in CHEM if c!="Fe2O3"]
            left=a[cols].to_numpy(float)
            right=b.loc[j,cols].to_numpy(float)
            shared=np.isfinite(left)&np.isfinite(right)
            agree=bool(shared.sum()>=5 and np.all(np.abs(left[shared]-right[shared])<=
                np.maximum(.001,.02*np.maximum(np.abs(left[shared]),np.abs(right[shared])))))
            if not (name.loc[j] or exact_id.loc[j] or (site.loc[j] and agree)):
                continue
            records.append({"source_dataset":row.source_dataset,"source_sample":row.source_sample,
                "comparison":origin,"comparison_position":str(j),"name_match":bool(name.loc[j]),
                "id_match":bool(exact_id.loc[j]),"site_match":bool(site.loc[j]),"shared_chemistry":int(shared.sum()),
                "chemistry_agreement":agree,"possible_duplicate":bool(exact_id.loc[j] or (site.loc[j] and (name.loc[j] or agree))),
                "comparison_sample":str(names.loc[j])})
    return records


def summarize(rows,edges,known_publications,domain,known_groups):
    r=rows.copy()
    chem=num(r,CHEM)
    r["chemical_predictors"]=chem.notna().sum(axis=1)
    r["oxide_sum"]=chem[list(OXIDES)].sum(axis=1,min_count=len(OXIDES))
    r["qc_diagnostic"]=r.oxide_sum.between(94,102)
    r["spatial_group"]=group(r)
    r["existing_group"]=r.spatial_group.isin(known_groups)
    r["academic_range"]=r.Li_icpms.between(domain["q025"],domain["q975"])
    r["positive_li"]=r.Li_icpms.gt(0)&np.isfinite(r.Li_icpms)
    r["li_uncensored"]=r.source_li_censoring.eq("")
    r["publication_overlap"]=[("doi:"+v.lower()) in known_publications for v in r.source_publication]
    blocked={(v["source_dataset"],v["source_sample"]) for v in edges if v["possible_duplicate"]}
    r["possible_duplicate"]=[(a,b) in blocked for a,b in zip(r.source_dataset,r.source_sample)]
    r["basic_qc"]=r.positive_li&r.li_uncensored&r.qc_diagnostic&r.chemical_predictors.ge(8)&r.Sample_type.eq("whole rock")&r.spatial_group.ne("")
    # Both sources require publication/method and inference compatibility review.
    # Macusani is a known GEOROC publication even if tokens do not resolve.
    r["publication_overlap"]|=r.source_dataset.eq(MAC)
    r["academic_review_candidate"]=r.basic_qc&r.academic_range&~r.possible_duplicate&~r.publication_overlap
    r["spatial_review_candidate"]=r.basic_qc&~r.possible_duplicate&~r.publication_overlap
    r["approved_for_training"]=False
    return r


