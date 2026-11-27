"""Paired single-scene versus temporal median using registered reflectance cubes."""
from __future__ import annotations

from datetime import date
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.reproducibility import canonical_json_hash
from lithiumscope.model_2.preparation.patch_contract import make_patch_contract, load_verified_patch
from lithiumscope.model_2.methodology.spectral.radiometric_features import apply_radiometry, extract_valid_spectral_features
from lithiumscope.tools.acquisition.sentinel2.registered_patch import verify_registered_mask, validate_interval
from lithiumscope.experiments.shared.paired_suite import Arm


def compare_cubes(cubes, dates):
    """At least three distinct dates, same predeclared window and common grid.

    Reference is the earliest qualifying scene, not chosen using lithium.
    Every retained pixel must have at least two valid dates in the composite.
    """
    if len(cubes)!=len(dates) or len(cubes)<3 or len(set(dates))!=len(dates):
        raise ValueError("Three or more distinct acquisition dates required")
    order=np.argsort(dates)
    values=np.asarray(cubes,dtype=np.float32)[order]
    if values.shape[1:]!=(6,32,32):
        raise ValueError("Expected registered six-band cubes")
    valid=np.isfinite(values).all(axis=1)
    values=np.where(valid[:,None,:,:],values,np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore",RuntimeWarning)
        median=np.nanmedian(values,axis=0)
    median[:,valid.sum(axis=0)<2]=np.nan
    return values[0], median


def temporal_features(cohort, acquisition_manifest, *, interval, root):
    start,end=validate_interval(interval)
    required={"sample_id","image_path","image_sha256","sidecar_sha256"}
    if not required<=set(acquisition_manifest) or not cohort.sample_id.is_unique:
        raise ValueError("Temporal acquisition manifest invalid")
    if set(acquisition_manifest.sample_id.astype(str))!=set(cohort.sample_id.astype(str)):
        raise ValueError("Temporal comparison must cover identical declared sample cohort")
    rows_single,rows_temporal,audit=[],[],[]
    for sample in cohort.itertuples(index=False):
        sid=str(sample.sample_id)
        subset=acquisition_manifest[acquisition_manifest.sample_id.astype(str)==sid]
        cubes,dates,scenes,hashes=[],[],[],[]
        grid=None
        for entry in subset.itertuples(index=False):
            path=(Path(root)/str(entry.image_path)).resolve()
            if not path.is_relative_to(Path(root).resolve()):
                raise ValueError("Temporal patch outside staged root")
            raw,meta,quality=load_verified_patch(path,make_patch_contract())
            if (quality["image_sha256"]!=entry.image_sha256 or quality["sidecar_sha256"]!=entry.sidecar_sha256
                    or str(meta["sample_id"])!=sid
                    or not np.allclose([meta["latitude"],meta["longitude"]],[sample.latitude,sample.longitude],rtol=0,atol=1e-8)):
                raise ValueError("Temporal source identity changed")
            verify_registered_mask(path,meta)
            day=date.fromisoformat(str(meta["datetime"])[:10])
            if not start<=day<=end or meta["scene_id"] in scenes:
                raise ValueError("Duplicate scene or out-of-window acquisition")
            geometry=meta["geometry"]
            key=(geometry["crs"],tuple(geometry["transform"]))
            if grid is not None and grid!=key:
                raise ValueError("Temporal grid mismatch")
            grid=key
            cubes.append(apply_radiometry(raw,geometry["radiometry"]))
            dates.append(day.isoformat());scenes.append(meta["scene_id"]);hashes.append(entry.image_sha256)
        single,median=compare_cubes(cubes,dates)
        rows_single.append(extract_valid_spectral_features(single,min_finite_fraction=.95))
        rows_temporal.append(extract_valid_spectral_features(median,min_finite_fraction=.95))
        audit.append({"sample_id":sid,"dates":dates,"scene_ids":scenes,"image_hashes":hashes,
                      "reference_date":min(dates),"composite_min_valid_dates":2})
    return pd.DataFrame(rows_single,index=cohort.index),pd.DataFrame(rows_temporal,index=cohort.index),audit


def temporal_arms(cohort, acquisitions, algorithms, *, interval, root, seed, mode, mask):
    single, composite, audit=temporal_features(cohort,acquisitions,interval=interval,root=root)
    policy={"seed":seed,"calibration":mode,"interval":interval,"acquisition_evidence_sha256":canonical_json_hash(audit)}
    arms=[]
    for algorithm in algorithms:
        ref=f"{algorithm}_single_date"
        arms.extend([Arm(ref,algorithm,None,single,{**policy,"representation":"earliest_qualifying_scene"},mask),
                     Arm(f"{algorithm}_temporal_median",algorithm,ref,composite,
                         {**policy,"representation":"median_reflectance_min_two_dates"},mask)])
    return arms,audit
