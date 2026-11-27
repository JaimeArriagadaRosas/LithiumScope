"""Historical public-document method scan; no specimen-level verification."""
from pathlib import Path
from lithiumscope.experiments.georoc.protocol import EXPERIMENT_ROOT
import hashlib,json
from lithiumscope.tools.acquisition.publications import fetch_public_pdf
from lithiumscope.results.atomic_exports import write_json
from pypdf import PdfReader


def main(*, acquire=False):
    if not acquire:
        return {"state": "preflight"}
    base=(EXPERIMENT_ROOT / "publication_sources")
    base.mkdir(parents=True,exist_ok=True)
    urls={
     "18365":"https://oceanrep.geomar.de/id/eprint/23604/1/jacques-csvz-cg2014onlinefirst.pdf",
     "17913":"https://oceanrep.geomar.de/21845/1/1-s2.0-S0016703713002895-main.pdf",
    }
    result=[]
    for ref,url in urls.items():
     try:
      p=base/(ref+".pdf")
      fetch_public_pdf(url,p,acquire=acquire)
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
    write_json(base/"paper_method_scan.json",result)

if __name__=="__main__":
    from lithiumscope.core.logging.session import run_log
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquire",action="store_true")
    if not parser.parse_args().acquire:
        print("Preflight: use --acquire to authorize network requests.")
        raise SystemExit(0)
    with run_log("simulations","publication_pdf_screen"):
        main(acquire=True)
