import hashlib
from pathlib import Path
import requests
out=Path("results/scientific_simulations/external_geochemistry_audit/georoc_andean_raw")
out.mkdir(parents=True,exist_ok=True)
for ident in (219128,219105,219094):
    dest=out/f"2026-09-PVFZCE_ANDEAN_ARC_part{(219128,219105,219094).index(ident)+1}.csv"
    if not dest.exists() or dest.stat().st_size<1000:
        with requests.get(f"https://data.goettingen-research-online.de/api/access/datafile/{ident}",stream=True,timeout=75) as resp:
            resp.raise_for_status()
            with dest.open("wb") as fh:
                for chunk in resp.iter_content(1024*1024):
                    if chunk:fh.write(chunk)
    print("DOWNLOADED",ident,dest.stat().st_size,hashlib.sha256(dest.read_bytes()).hexdigest(),flush=True)
    if ident==219128:
        with dest.open("r",encoding="utf-8-sig",errors="replace") as f:
            for i in range(3):
                print("HEAD",i,f.readline()[:1000],flush=True)

