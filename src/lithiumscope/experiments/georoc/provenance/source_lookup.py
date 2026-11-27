"""Find legitimate article/supplement URLs for the prioritized 10 GEOROC references."""
from pathlib import Path
from lithiumscope.experiments.georoc.protocol import EXPERIMENT_ROOT
import pandas as pd
from lithiumscope.tools.acquisition.publications import lookup_doi
from lithiumscope.results.atomic_exports import write_json,write_table
BASE=EXPERIMENT_ROOT


def main(*, acquire=False):
    if not acquire:
        return {"state": "preflight"}
    frame=pd.read_csv(BASE/"publication_priority_top10.csv")
    records=[]
    for row in frame.itertuples():
     doi=str(row.DOI).strip().lower()
     if doi=="nan":continue
     try:
      data=lookup_doi(doi,acquire=acquire)
      best=data.get("best_oa_location") or {}
      entry={"citation_number":row.citation_number,"doi":doi,
       "title":data.get("title",""),"is_oa":data.get("open_access",{}).get("is_oa"),
       "oa_pdf":best.get("pdf_url"),"oa_landing":best.get("landing_page_url"),
       "version":best.get("version"),"location_count":len(data.get("locations",[])),
       "openalex_id":data.get("id"),"method_verified":False}
      records.append(entry)
      print("RESOLVED",row.citation_number,entry["is_oa"],entry["oa_pdf"],flush=True)
     except Exception as exc:
      records.append({"citation_number":row.citation_number,"doi":doi,"lookup_error":str(exc)[:120]})
      print("ERROR",row.citation_number,str(exc)[:95],flush=True)
    write_table(BASE/"top10_open_access_locations.csv",pd.DataFrame(records))
    write_json(BASE/"top10_source_lookup_manifest.json",{"lookups":len(records),
     "open_access":sum(x.get("is_oa") is True for x in records),
     "method_verified":0,"note":"Index URLs do not prove specimen-level lithium analytical method."})

if __name__=="__main__":
    from lithiumscope.core.logging.session import run_log
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquire",action="store_true")
    if not parser.parse_args().acquire:
        print("Preflight: use --acquire to authorize network requests.")
        raise SystemExit(0)
    with run_log("simulations","publication_source_lookup"):
        main(acquire=True)
