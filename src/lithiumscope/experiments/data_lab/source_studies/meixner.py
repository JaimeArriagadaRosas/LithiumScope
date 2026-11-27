"""Meixner original workbook and target-independent sample identity review."""
import numpy as np
import pandas as pd
from lithiumscope.experiments.data_lab.screening.candidates import CHEM, sample, num

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


