"""Screen author+year citation overlap with all Mamani09 original sources.

A bibliographic screen is a risk signal; it cannot prove independence.
"""
from __future__ import annotations
import re,json
from pathlib import Path
import pandas as pd
BASE=Path("results/scientific_simulations/external_geochemistry_audit")
SRC=Path(r"D:\Descargas\LithiumScope\data\raw\model_1\Mamani09_Table_DR2.csv")
def norm(s):
    return re.sub(r"[^a-z0-9]+"," ",str(s).casefold()).strip()
def main():
    mamani=pd.read_csv(SRC,encoding="latin1",low_memory=False)
    authors=list(mamani["Geochemistry Reference"].dropna().astype(str).unique())
    # Mamani citations may include several sources with semicolon separators.
    keys=set()
    for item in authors:
        for fragment in item.split(";"):
            v=norm(fragment)
            # Last name plus year is a *risk flag*, not a proof of sample identity.
            match=re.search(r"([a-z]+)\s+(?:and\s+[a-z]+\s+)?(?:et\s+al\s+)?(19\d{2}|20\d{2})",v)
            if match:keys.add(match.group(1)+"|"+match.group(2))
    refs=pd.read_csv(BASE/"all_reviewable_citations.csv",low_memory=False)
    def check(row):
        year=re.search(r"(19\d{2}|20\d{2})",str(row.YEAR))
        auth=re.search(r"[A-Za-zÀ-ÿ]+",str(row.AUTHORS))
        if not year or not auth:return "UNKNOWN"
        candidate=norm(auth.group(0))+"|"+year.group(1)
        return "AUTHOR_YEAR_OVERLAP" if candidate in keys else "NO_AUTHOR_YEAR_MATCH"
    refs["mama_author_year_screen"]=refs.apply(check,axis=1)
    refs["reference_independent_approved"]=False
    refs.to_csv(BASE/"citation_mamani_overlap_screen.csv",index=False)
    summary={"citations_checked":len(refs),"mama_reference_keys":len(keys),
        "flag_counts":refs.mama_author_year_screen.value_counts().to_dict(),
        "reference_independence_approved":0,
        "limitation":"Author and year matching cannot rule out common samples, derived data, or different citations of the same original measurement."}
    (BASE/"citation_mamani_overlap_screen.json").write_text(json.dumps(summary,indent=2),encoding="utf8")
    print("CITATION_OVERLAP",json.dumps(summary,ensure_ascii=False),flush=True)
if __name__=="__main__":main()
