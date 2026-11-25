"""Read GEOROC Dataverse candidate fields without asserting independence."""
from __future__ import annotations
import numpy as np
import pandas as pd

REQUIRED=("LATITUDE MIN","LONGITUDE MIN","LI(PPM)","MATERIAL","SAMPLE NAME","CITATIONS","LOCATION")


def whole_rock_candidates(frame,source_file):
    if not set(REQUIRED)<=set(frame):
        raise ValueError("GEOROC source schema incomplete")
    lat,lon,li=(pd.to_numeric(frame[c],errors="coerce") for c in REQUIRED[:3])
    material=frame["MATERIAL"].fillna("").astype(str)
    mask=np.isfinite(li)&li.ge(0)&lat.between(-90,90)&lon.between(-180,180)&material.str.contains(r"\bWR\b|WHOLE ROCK",case=False,regex=True)
    selected=frame.loc[mask]
    result=pd.DataFrame({"source_file":source_file,"sample_name":selected["SAMPLE NAME"].astype(str),
        "latitude":lat[mask],"longitude":lon[mask],"li_ppm":li[mask],"material":selected.MATERIAL.astype(str),
        "citation_ids":selected.CITATIONS.astype(str),"location":selected.LOCATION.astype(str)})
    return result.reset_index(drop=True),{"all_rows":len(frame),"li_numeric":int(np.isfinite(li).sum()),"candidate_whole_rock":len(result)}
