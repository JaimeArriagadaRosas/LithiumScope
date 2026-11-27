"""Coordinate alias assessment without guessing datum, precision or axis corrections."""
from __future__ import annotations

import numpy as np
import pandas as pd

LONGITUDE=("Longitude","Longitude (X)","Logintude (X)","longitude")
LATITUDE=("Latitude","Latitude (Y)","latitude")
ALIAS_TOLERANCE_DEG=1e-5  # Agreement tolerance, not measured coordinate accuracy.
CRS_COLUMNS=("source_coordinate_crs","coordinate_crs","source_crs","CRS","crs","datum","Datum")
PRECISION_COLUMNS=("source_coordinate_precision","coordinate_precision","precision_m",
                   "coordinate_uncertainty_m","Latitude_min","Latitude_max","Longitude_min","Longitude_max")
UNKNOWN={"","unknown","none","nan","na","n/a","not reported","<na>"}
WGS84={"epsg:4326","4326","wgs84","wgs 84","wgs_84","ogc:crs84","crs84"}


def _axis(frame,names,limit,*,longitude=False):
    columns=[c for c in names if c in frame]
    if not columns:
        empty=pd.Series(np.nan,index=frame.index,dtype=float)
        false=pd.Series(False,index=frame.index,dtype=bool)
        return empty,false,false.copy(),empty.copy(),pd.Series("",index=frame.index,dtype=str)
    parsed=pd.DataFrame({c:pd.to_numeric(frame[c],errors="coerce").astype(float) for c in columns},index=frame.index)
    present=pd.DataFrame({c:frame[c].notna() & ~frame[c].astype(str).str.strip().str.lower().isin(UNKNOWN)
                          for c in columns},index=frame.index)
    valid=np.isfinite(parsed)&parsed.ge(-limit)&parsed.le(limit)
    invalid=(present&~valid).any(axis=1)
    usable=parsed.where(valid)
    chosen=usable.bfill(axis=1).iloc[:,0]
    source=pd.Series("",index=frame.index,dtype=str)
    for name in columns:
        source=source.mask(source.eq("")&valid[name],name)
    from itertools import combinations
    differences=[]
    for left,right in combinations(columns,2):
        diff=(usable[left]-usable[right]).abs()
        if longitude:
            diff=diff.where(diff<=180,360-diff)
        differences.append(diff)
    maximum=pd.concat(differences,axis=1).max(axis=1) if differences else chosen.where(chosen.isna(),0.)
    maximum=maximum.where(maximum.notna()|chosen.isna(),0.)
    if longitude:
        chosen=chosen.mask(chosen.eq(180),-180.)
    return chosen,maximum.gt(ALIAS_TOLERANCE_DEG),invalid,maximum,source


def _crs_evidence(frame):
    reported=pd.DataFrame({c:frame[c].astype("string").fillna("").str.strip().str.lower()
                           for c in CRS_COLUMNS if c in frame},index=frame.index)
    known=~reported.isin(UNKNOWN)
    unsupported=(known&~reported.isin(WGS84)).any(axis=1)
    declared=known.any(axis=1)
    state=pd.Series("unknown_not_certified",index=frame.index,dtype=str)
    state.loc[declared&~unsupported]="reported_wgs84"
    state.loc[unsupported]="requires_crs_review"
    return state


def assess_coordinate_aliases(frame,*,original=None):
    if not frame.index.is_unique or not frame.columns.is_unique:
        raise ValueError("Coordinate assessment requires unique rows and columns")
    original=frame if original is None else original
    if not original.index.equals(frame.index):
        raise ValueError("Original coordinate aliases are not aligned")
    lon,lon_conf,lon_invalid,lon_diff,lon_source=_axis(frame,LONGITUDE,180,longitude=True)
    lat,lat_conf,lat_invalid,lat_diff,lat_source=_axis(frame,LATITUDE,90)
    crs=_crs_evidence(original)
    missing=lon.isna()|lat.isna()
    conflict=lon_conf|lat_conf
    invalid=lon_invalid|lat_invalid
    status=pd.Series("valid_numeric_coordinates",index=frame.index,dtype=str)
    status.loc[missing]="missing_coordinate"
    status.loc[invalid]="invalid_alias"
    status.loc[conflict]="conflicting_aliases"
    status.loc[crs.eq("requires_crs_review")]="requires_crs_review"
    # Detect obvious swapped axes, but never swap them automatically.
    swap=pd.Series(False,index=frame.index)
    for y in LATITUDE:
        for x in LONGITUDE:
            if x in frame and y in frame:
                a,b=pd.to_numeric(frame[y],errors="coerce"),pd.to_numeric(frame[x],errors="coerce")
                swap |= a.abs().gt(90)&a.abs().le(180)&b.abs().le(90)
    status.loc[swap]="possible_axis_swap"
    evidence=pd.DataFrame({"row_index":frame.index.astype(str),"coordinate_status":status,
                           "longitude_source_alias":lon_source,"latitude_source_alias":lat_source,
                           "longitude_candidate":lon,"latitude_candidate":lat,
                           "longitude_max_alias_difference_deg":lon_diff,
                           "latitude_max_alias_difference_deg":lat_diff,
                           "longitude_alias_conflict":lon_conf,"latitude_alias_conflict":lat_conf,
                           "invalid_alias_present":invalid,"possible_axis_swap":swap,
                           "crs_status":crs},index=frame.index)
    for name in (*LONGITUDE,*LATITUDE,*CRS_COLUMNS,*PRECISION_COLUMNS,"source_dataset","source_sample","Sample"):
        if name in original:
            evidence[f"original_{name}"]=original[name]
    for axis in ("Latitude","Longitude"):
        lo,hi=f"{axis}_min",f"{axis}_max"
        if {lo,hi}<=set(original):
            evidence[f"reported_{axis.lower()}_span_deg"]=(pd.to_numeric(original[hi],errors="coerce")-
                                                            pd.to_numeric(original[lo],errors="coerce")).abs()
    # Bounds are a reported extent proxy only; not a known sampling error or datum.
    usable=status.eq("valid_numeric_coordinates")
    result=frame.copy()
    if any(c in frame for c in LONGITUDE):
        result["Longitude"]=lon.where(usable)
    if any(c in frame for c in LATITUDE):
        result["Latitude"]=lat.where(usable)
    return result,evidence


def coordinate_summary(evidence):
    status=evidence.coordinate_status
    valid=status.eq("valid_numeric_coordinates")
    return {"alias_tolerance_degrees":ALIAS_TOLERANCE_DEG,
            "coordinate_status_counts":{str(k):int(v) for k,v in status.value_counts().items()},
            "unresolved_coordinate_rows":int((~valid).sum()),
            "conflicting_alias_rows":int((evidence.longitude_alias_conflict|evidence.latitude_alias_conflict).sum()),
            "invalid_alias_rows":int(evidence.invalid_alias_present.sum()),
            "possible_axis_swap_rows":int(evidence.possible_axis_swap.sum()),
            "unknown_crs_rows":int(evidence.crs_status.eq("unknown_not_certified").sum()),
            "reported_wgs84_rows":int(evidence.crs_status.eq("reported_wgs84").sum()),
            "requires_crs_review_rows":int(evidence.crs_status.eq("requires_crs_review").sum()),
            "coordinate_accuracy_certified":False}
