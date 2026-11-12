"""Find legitimate article/supplement URLs for the prioritized 10 GEOROC references."""
from pathlib import Path
import requests,pandas as pd,json,time
BASE=Path("results/scientific_simulations/external_geochemistry_audit")
frame=pd.read_csv(BASE/"publication_priority_top10.csv")
records=[]
for row in frame.itertuples():
 doi=str(row.DOI).strip().lower()
 if doi=="nan":continue
 try:
  res=requests.get("https://api.openalex.org/works/https://doi.org/"+doi,timeout=16)
  res.raise_for_status();data=res.json()
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
pd.DataFrame(records).to_csv(BASE/"top10_open_access_locations.csv",index=False)
(BASE/"top10_source_lookup_manifest.json").write_text(json.dumps({"lookups":len(records),
 "open_access":sum(x.get("is_oa") is True for x in records),
 "method_verified":0,"note":"Index URLs do not prove specimen-level lithium analytical method."},indent=2),encoding="utf8")
