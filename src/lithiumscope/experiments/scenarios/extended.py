"""Independent seasonal and shifted-patch stress plan; coordinates do not imply Li labels."""
from __future__ import annotations
from pathlib import Path
import json
import pandas as pd

from lithiumscope.experiments.acquisition.acquisition import acquire_pilot, PILOT_LOCATIONS
from lithiumscope.experiments.scenarios.batch import run_m2_batch, summarize
from lithiumscope.experiments.reporting.reporting import compare_geographies


def experiment_locations() -> dict[str, tuple[float, float]]:
    # Approximate shifts of 300-500 metres, never reusing the original site label.
    positions = {}
    for name, (lat, lon) in PILOT_LOCATIONS.items():
        positions[f"{name}_east"] = (lat, lon + 0.004)
        positions[f"{name}_north"] = (lat + 0.004, lon)
    return positions


def run_extended(destination: Path, bundle: dict, *, year: int = 2025) -> dict:
    summer = destination / "summer"
    winter = destination / "winter"
    shifted = destination / "shifted"
    # Distinct directory per season avoids overwriting the main pilot's scene cache.
    acquire_pilot(summer, years=(year,), locations=PILOT_LOCATIONS,
                  date_window=("01-01", "03-31"), max_patches=4)
    acquire_pilot(winter, years=(year,), locations=PILOT_LOCATIONS,
                  date_window=("06-01", "08-31"), max_patches=4)
    acquire_pilot(shifted, years=(year,),
                  locations=experiment_locations(), max_patches=8)
    manifests=[folder/"manifest.csv" for folder in (summer,winter,shifted)]
    records=[]
    for label, file in zip(("summer","winter","shifted"), manifests):
        if file.is_file():
            table=pd.read_csv(file)
            if not table.empty:
                table["experiment_group"]=label
                records.append(table)
    combined=destination/"extended_manifest.csv"
    if records:
        pd.concat(records,ignore_index=True).to_csv(combined,index=False)
    else:
        pd.DataFrame(columns=["image_id","scene_id","year","path","status"]).to_csv(combined,index=False)
    output=run_m2_batch(bundle,combined,destination/"batch")
    report=summarize(output,output.parent)
    compare_geographies(output,output.parent)
    overview={"manifest":str(combined),"batch":str(output),"summary":str(report),
              "groups":{k:int((df["experiment_group"]==k).sum())
                        for k in ("summer","winter","shifted")
                        for df in [pd.read_csv(combined)]} if records else {},
              "interpretation":"Unlabelled imagery: robustness only, never independent chemical accuracy."}
    (destination/"extended_report.json").write_text(json.dumps(overview,indent=2),encoding="utf-8")
    return overview
