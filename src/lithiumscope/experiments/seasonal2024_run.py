"""Reproducible additional seasonal sampling in 2024; no model fitting."""
from pathlib import Path
import pandas as pd
from lithiumscope.experiments.acquisition import acquire_pilot, PILOT_LOCATIONS
from lithiumscope.experiments.batch import run_m2_batch, summarize
from lithiumscope.persistence.load_model import load_latest_model

base=Path("results/scientific_simulations/seasonal_2024")
records=[]
for label,dates in (("summer",("01-01","03-31")),("winter",("06-01","08-31"))):
    places={name+"_"+label:position for name,position in PILOT_LOCATIONS.items()}
    print("PERIOD",label,flush=True)
    manifest,failures=acquire_pilot(base/label,years=(2024,),locations=places,max_patches=4,timeout_seconds=45,date_window=dates)
    table=pd.read_csv(manifest) if manifest.exists() else pd.DataFrame()
    print("PERIOD_RESULT",label,len(table),table.status.value_counts().to_dict() if not table.empty else {},flush=True)
    if not table.empty:
        table["experiment_group"]=label
        records.append(table)
if not records:
    raise RuntimeError("No acquired scene metadata")
combined=base/"seasonal_manifest.csv"
pd.concat(records,ignore_index=True).to_csv(combined,index=False)
bundle,_=load_latest_model("model_2")
result=run_m2_batch(bundle,combined,base/"batch")
print("RESULT",result,flush=True)
print("SUMMARY",summarize(result,result.parent),flush=True)
print("STATUSES",pd.read_csv(result).status.value_counts().to_dict(),flush=True)
