"""Acquire a four-location external-candidate imagery PILOT only (no accuracy claims)."""
from pathlib import Path
from lithiumscope.experiments.data_lab.source_studies.georoc.protocol import EXPERIMENT_ROOT
import pandas as pd
from lithiumscope.tools.acquisition.sentinel2.pilot import acquire_pilot
from lithiumscope.experiments.scenarios.batch import run_m2_batch,summarize
from lithiumscope.persistence.load_model import load_latest_model



def main():
    base=EXPERIMENT_ROOT
    candidates=pd.read_csv(base/"georoc_provenance_screen.csv")
    references=["27313","29268","29484","23382"]
    chosen=[]
    for reference in references:
        matches=candidates.loc[(candidates.country=="CHILE")&
            (candidates.citation_key==reference)&
            (candidates.nearest_training_km>=100)&
            (~candidates.duplicate_identifier_location)]
        if matches.empty:
            raise ValueError(f"No eligible pilot candidate for reference {reference}")
        # Preserve historical ordering explicitly, not an implicit index error.
        chosen.append(matches.iloc[0])
    pilot=pd.DataFrame(chosen)
    pilot["image_id"]=["georoc_"+reference for reference in references]
    pilot.to_csv(base/"pilot_target_metadata.csv",index=False)
    locations={row.image_id:(row.latitude,row.longitude) for row in pilot.itertuples()}
    manifest,_=acquire_pilot(base/"pilot_sentinel",years=(2025,),locations=locations,max_patches=4,timeout_seconds=45)
    data=pd.read_csv(manifest)
    print("PILOT_ACQUISITION",data.status.value_counts().to_dict(),flush=True)
    if len(data):
        model,_=load_latest_model("model_2")
        output=run_m2_batch(model,manifest,base/"pilot_sentinel"/"batch")
        print("PILOT_BATCH",output,flush=True)
        print("PILOT_SUMMARY",summarize(output,output.parent),flush=True)


if __name__ == "__main__":
    import argparse
    from lithiumscope.core.logging.session import run_log
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquire",action="store_true")
    if parser.parse_args().acquire:
        with run_log("simulations","georoc_imagery_pilot"):
            main()
    else:
        print("Preflight: use --acquire to authorize acquisition.")
