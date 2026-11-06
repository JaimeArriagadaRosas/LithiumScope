"""One verifiable geochemistry candidate, NOT eligible without full independence/method audit."""
from pathlib import Path
from datetime import datetime,timezone
from math import radians, sin, cos, asin, sqrt
import hashlib,html,json,re
import pandas as pd
import requests


URL="https://georoc.mpch-mainz.gwdg.de/georoc/navdat_sampleinfo.asp?unique_id=302366"
TRAIN=Path(r"D:\Descargas\LithiumScope\data\processed\model_2\training_manifest.csv")
OUT=Path("results/scientific_simulations/external_geochemistry_audit")


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    response=requests.get(URL,timeout=25)
    response.raise_for_status()
    text=html.unescape(re.sub(r"<[^>]+>"," ",response.text))
    text=re.sub(r"\s+"," ",text)
    coords=re.search(r"Latitude\s*(-?\d+(?:\.\d+)?)\s*\|\s*Longitude\s*(-?\d+(?:\.\d+)?)",text)
    li=re.search(r"\bLI\s+(\d+(?:\.\d+)?)\s+PPM\b",text)
    rock=re.search(r"Analyzed:\s*WHOLE ROCK",text)
    assert coords and li and rock,"Missing required evidence in GEOROC page"
    lat,lon=map(float,coords.groups())
    lithium=float(li.group(1))
    training=pd.read_csv(TRAIN)
    n=len(training)
    lat1,lon1=map(radians,(lat,lon))
    lat2=training.latitude.map(radians)
    lon2=training.longitude.map(radians)
    dlat=lat2-lat1;dlon=lon2-lon1
    a=dlat.map(lambda v:sin(v/2)**2)+cos(lat1)*lat2.map(cos)*dlon.map(lambda v:sin(v/2)**2)
    # Vectorized pandas inputs, result in kilometers.
    distances=6371.0*2*a.map(lambda v:asin(sqrt(min(1,max(0,v)))))
    sample={
        "sample_id":"GEOROC-302366",
        "material":"whole_rock","latitude":lat,"longitude":lon,
        "Li_value":lithium,"Li_unit":"ppm",
        "analytical_method":"",
        "source":URL,
        "sample_date":"",
        "independent_of_training":"unverified",
        "source_sample_name":"385",
        "reference":"Assimilation of the plutonic roots of the Andean arc controls variations in U-series disequilibria at Volcan Llaima, Chile; EPSL 303 (2011) 37-47",
        "nearest_training_distance_km":round(float(distances.min()),3),
        "n_training_within_50km":int((distances<50).sum()),
        "n_training_samples":n,
        "eligibility":"REQUIRES_METHOD_AND_OVERLAP_AUDIT",
    }
    pd.DataFrame([sample]).to_csv(OUT/"georoc_candidates.csv",index=False)
    report={
        "status":"SOURCE_VERIFIED_CANDIDATE_ONLY",
        "retrieved_utc":datetime.now(timezone.utc).isoformat(),
        "source_url":URL,
        "source_sha256":hashlib.sha256(response.content).hexdigest(),
        "training_manifest_sha256":hashlib.sha256(TRAIN.read_bytes()).hexdigest(),
        "candidate_count":1,
        "validated_external_labels":0,
        "nearest_training_distance_km":sample["nearest_training_distance_km"],
        "method_missing":True,
        "independence_not_proven":True,
        "next_steps":["Verify measurement method in cited article",
                      "Verify sample-level provenance and overlap with training source",
                      "Ensure model-specific feature and label compatibility",
                      "Build a multi-region holdout with enough independent labels"],
    }
    (OUT/"audit_manifest.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print("AUDIT",json.dumps(report,indent=2),flush=True)


if __name__=="__main__":main()

