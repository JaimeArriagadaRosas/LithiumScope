"""Export M1 EDA tables and exact preparation lineage."""
from __future__ import annotations

import pandas as pd


def write_descriptive_tables(tables, prepared, directory):
    for name,frame in tables.items():
        frame.to_csv(directory/f"{name}.csv",index=name=="spearman_correlations")
    prepared.audit.to_frame().to_csv(directory/"preparation_stages.csv",index=False)
    pd.DataFrame(prepared.audit.source_records).to_csv(directory/"stage_sources.csv",index=False)
    prepared.audit.exclusions_frame().to_csv(directory/"stage_exclusions.csv",index=False)
    pd.DataFrame(prepared.audit.column_records).to_csv(directory/"stage_column_missingness.csv",index=False)
    if prepared.coordinate_evidence is not None:
        prepared.coordinate_evidence.to_csv(directory/"coordinate_alias_audit.csv",index=False)
    reasons=[]
    for stage,attr in (("05_target_validity","target_exclusions"),("06_quality_control","qc_exclusions")):
        reasons.extend({"stage":stage,**record} for record in prepared.frame.attrs.get(attr,[]))
    pd.DataFrame(reasons,columns=["stage","row_index","column","original","reason"]).to_csv(directory/"exclusion_reasons.csv",index=False)
    pd.DataFrame(prepared.audit.censoring,columns=["row_index","source","column","original","censor_type","limit","substituted"]).to_csv(directory/"censoring.csv",index=False)
