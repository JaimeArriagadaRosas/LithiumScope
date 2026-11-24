"""Retrospective source reconciliation; no canonical data mutation or geologic inference."""
from __future__ import annotations

import re
import numpy as np
import pandas as pd

from lithiumscope.model_1.training.preprocessing.tabular import OXIDES, TRACE_ELEMENTS

CHEMISTRY = [*OXIDES, *TRACE_ELEMENTS, "SUM (no water)"]
KEYS = ["_sample", "_li", "_latitude", "_longitude"]
FIELDS = ["source_unique_id", "source_rock_class", "source_publication",
          "source_location", "source_li_unit", "source_li_method"]
UNKNOWN = {"", "unknown", "not given", "not reported", "not specified", "n/a", "na", "none"}


def _keys(frame):
    result=frame.copy()
    result["_sample"]=result.source_sample.fillna("").astype(str).str.strip()
    for source,target in (("Li_icpms","_li"),("Latitude","_latitude"),("Longitude","_longitude")):
        result[target]=pd.to_numeric(result[source],errors="coerce").round(8)
    return result


def _valid_keys(frame):
    return frame._sample.ne("") & np.isfinite(frame[KEYS[1:]]).all(axis=1)


def reconcile_georoc(historical, source):
    """Require identity, chemistry, missingness and Fe convention; never break ties."""
    required={"source_sample","Li_icpms","Latitude","Longitude","Fe_total_provenance",*CHEMISTRY}
    if not required<=set(historical) or not required.union(FIELDS)<=set(source):
        raise ValueError("Missing chemical provenance fields")
    if not historical.index.is_unique:
        raise ValueError("Historical source rows must have unique identity")
    if source.source_unique_id.isna().any() or source.source_unique_id.astype(str).duplicated().any():
        raise ValueError("Missing or duplicate original GEOROC IDs")
    left=_keys(historical)
    left["_historical_row"]=left.index
    right=_keys(source)
    valid=_valid_keys(left)
    right=right.loc[_valid_keys(right)]
    columns=[*KEYS,*CHEMISTRY,"Fe_total_provenance"]
    joined=left.loc[valid,["_historical_row",*columns]].merge(
        right[[*columns,*FIELDS]],on=KEYS,how="left",suffixes=("_historical","_source"))
    same=joined.source_unique_id.notna().to_numpy(copy=True)
    for name in CHEMISTRY:
        a=pd.to_numeric(joined[name+"_historical"],errors="coerce").to_numpy(float)
        b=pd.to_numeric(joined[name+"_source"],errors="coerce").to_numpy(float)
        # Missing is not equivalent to a measured 0. Invalid numeric strings
        # must be handled upstream and cannot count as chemical evidence.
        same &= np.isclose(a,b,rtol=1e-12,atol=1e-10,equal_nan=True)
        for side, numeric in (("historical",a),("source",b)):
            original=joined[name+"_"+side]
            same &= ~((original.notna() & original.astype(str).str.strip().ne("")).to_numpy() & ~np.isfinite(numeric))
    fe_left=joined.Fe_total_provenance_historical
    fe_right=joined.Fe_total_provenance_source
    same &= fe_left.eq(fe_right).fillna(False).to_numpy() | (fe_left.isna()&fe_right.isna()).to_numpy()
    joined["_chemical_match"]=same
    records=[]
    for idx,group in joined.groupby("_historical_row",sort=False):
        possible=group.loc[group.source_unique_id.notna()]
        matches=group.loc[group._chemical_match]
        record={"source_row":idx,"identity_candidates":len(possible),"chemical_matches":len(matches),
                "identity_candidate_ids":";".join(possible.source_unique_id.astype(str)),
                "chemical_match_ids":";".join(matches.source_unique_id.astype(str)),
                "state":"unique" if len(matches)==1 else "ambiguous" if len(matches)>1 else
                        "chemistry_mismatch" if len(possible) else "no_identity_match"}
        if len(matches)==1:
            record.update({c:matches.iloc[0][c] for c in FIELDS})
        records.append(record)
    for idx in left.index[~valid]:
        records.append({"source_row":idx,"identity_candidates":0,"chemical_matches":0,"state":"missing_identity_key"})
    result=pd.DataFrame(records).sort_values("source_row").reset_index(drop=True)
    if len(result)!=len(historical) or result.source_row.duplicated().any():
        raise ValueError("Historical row identity lost")
    return result


def reported_rock_label(value):
    """Remove numeric reference markers; only restore a single reported label.

    Multiple conflicting names and unreported values remain missing. Do not
    infer a lithology from chemistry, coordinate, source publication or Li.
    """
    if pd.isna(value):
        return None
    text=re.sub(r"\[\d+\]","",str(value))
    labels={re.sub(r"\s+"," ",v.strip().casefold()) for v in text.split("/")}
    known=labels-UNKNOWN
    return next(iter(known)) if len(known)==1 else None


def oxide_balance(frame):
    """Compare observed sum conventions, not a new quality-control rule."""
    oxides=frame[OXIDES].apply(pd.to_numeric,errors="coerce")
    feo=pd.to_numeric(frame["FeO"],errors="coerce")
    reported=pd.to_numeric(frame["SUM (no water)"],errors="coerce")
    sum10=oxides.sum(axis=1,min_count=1)
    total=sum10+feo.fillna(0)
    return pd.DataFrame({
        "reported_sum":reported,"oxide_count":oxides.notna().sum(axis=1),
        "sum_10_oxides":sum10,"reported_feo":feo,"sum_10_plus_reported_feo":total,
        "difference_reported_minus_10":reported-sum10,
        "difference_reported_minus_10_plus_feo":reported-total,
        "matches_10_within_0_05":(reported-sum10).abs()<=.05,
        "matches_10_plus_feo_within_0_05":(reported-total).abs()<=.05,
    },index=frame.index)
