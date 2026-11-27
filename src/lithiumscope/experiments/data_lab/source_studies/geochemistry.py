"""Review-only chemical measurement parsing; no training cohort dependency."""
import re
import numpy as np
import pandas as pd
from lithiumscope.datasets.validation.chemistry.analytical import parse_analytical_value
from lithiumscope.experiments.data_lab.screening.candidates import CHEM

ELEMENTS=set("Be B F Sc Cr Co Ni Cu Zn Ga Ge As Se Y Mo Ag Cd In Sn Sb Te La Ce Pr Nd Sm Eu Gd Tb Dy Ho Er Tm Yb Lu W Re Tl Bi S Cl Br Rb Cs Nb Ta Pb Ba Sr Zr V Hf Th U".lower().split())
# Elements not already represented by the canonical trace predictor set.
EXTRA=ELEMENTS-set("th u rb cs nb ta pb ba sr zr v hf".split())

def measurement_map(header):
    found={}
    for name in header:
        match=re.fullmatch(r"\s*([A-Za-z]{1,2})\s*\(\s*(ppm|ppb)\s*\)\s*",str(name),re.I)
        if not match or match[1].lower() not in EXTRA:
            continue
        key=match[1].capitalize()
        found.setdefault(key,[]).append((name,match[2].lower()))
    # Never choose arbitrarily between multiple fields for the same element.
    return {key:items[0] for key,items in found.items() if len(items)==1}

def numeric_ppm(values,unit):
    if unit not in {"ppm","ppb"}:
        raise ValueError("Unsupported measurement unit")
    parsed=values.map(lambda v:parse_analytical_value(v,decimal_comma=False)[0])
    x=pd.to_numeric(parsed,errors="coerce").astype(float)
    return x.where(np.isfinite(x)&x.ge(0))*(.001 if unit=="ppb" else 1.)


def raw_view(chunk,mapping):
    """Do not call harmonize_chunk: it drops rows without Li before identity review."""
    frame=chunk.rename(columns=mapping).copy()
    for c in (*CHEM,"Li_icpms"):
        if c in frame:
            parsed=frame[c].map(lambda v:parse_analytical_value(v,decimal_comma=False)[0])
            # Analytical parsing handles censoring, but leaves ordinary strings
            # unchanged. Numeric conversion is a separate required step.
            values=pd.to_numeric(parsed,errors="coerce").astype(float)
            frame[c]=values.where(np.isfinite(values)&values.ge(0))
    for axis in ("Latitude","Longitude"):
        if axis not in frame and axis+"_min" in frame:
            low=pd.to_numeric(frame[axis+"_min"],errors="coerce")
            high=pd.to_numeric(frame.get(axis+"_max",low),errors="coerce").fillna(low)
            frame[axis]=(low+high)/2
    return frame


