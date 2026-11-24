from pathlib import Path
import requests,hashlib,json
from pypdf import PdfReader
base=Path("results/scientific_simulations/external_geochemistry_audit/publication_sources")
base.mkdir(parents=True,exist_ok=True)
urls={
 "18365":"https://oceanrep.geomar.de/id/eprint/23604/1/jacques-csvz-cg2014onlinefirst.pdf",
 "17913":"https://oceanrep.geomar.de/21845/1/1-s2.0-S0016703713002895-main.pdf",
}
result=[]
for ref,url in urls.items():
 try:
  p=base/(ref+".pdf")
  if not p.exists():
   s=requests.get(url,timeout=40)
   s.raise_for_status()
   if not s.content.startswith(b"%PDF"):raise ValueError("Not PDF")
   p.write_bytes(s.content)
  pdf=PdfReader(str(p))
  hits=[]
  for n,page in enumerate(pdf.pages):
   s=page.extract_text() or ""
   for needle in ("ICP-MS","ICP–MS","lithium","Li concentrations","trace element","LA-ICP"):
    if needle.lower() in s.lower():
     i=s.lower().find(needle.lower())
     hits.append({"page":n+1,"term":needle,"excerpt":" ".join(s[max(0,i-80):i+170].split())[:250]})
   if len(hits)>=14:break
  item={"reference_id":ref,"url":url,"pages":len(pdf.pages),"sha256":hashlib.sha256(p.read_bytes()).hexdigest(),"method_mentions":hits[:14],"specific_li_assay_verified":False}
  result.append(item)
  print("DOC",ref,"pages",len(pdf.pages),"hits",hits[:4],flush=True)
 except Exception as e:result.append({"reference_id":ref,"error":str(e)[:160]});print("ERROR",ref,str(e)[:160],flush=True)
(base/"paper_method_scan.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf8")
