"""Predeclared synthetic stress scenarios; not laboratory uncertainty estimates."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from lithiumscope.core.scientific.regression_metrics import regression_metrics
from lithiumscope.model_1.preparation.feature_engineering import add_geochemical_features
from lithiumscope.model_1.training.preprocessing.tabular import OXIDES, TRACE_ELEMENTS, DERIVED

POLICY = "physical_synthetic_sensitivity_v1"
SCENARIOS = {"oxide_relative": (.01, .05), "trace_relative": (.01, .05),
             "missingness": (.05, .15), "spatial_m": (100., 500.)}
SEED_OFFSETS = (0, 1)


@dataclass(frozen=True)
class StressInput:
    frame: pd.DataFrame
    eligible_cells: int
    changed_cells: int
    rejected_cells: int
    preexisting_invalid_cells: int
    eligible_rows: int


def _chemical(frame, names, upper):
    cols = [c for c in names if c in frame]
    values = frame[cols].to_numpy(dtype=float, copy=True)
    valid = np.isfinite(values) & (values >= 0) & (values <= upper)
    invalid = ~np.isnan(values) & ~valid
    return cols, values, valid, int(invalid.sum())


def _relative(frame, names, upper, magnitude, rng, *, preserve_sum):
    cols, values, valid, invalid = _chemical(frame, names, upper)
    candidate = np.where(valid, values * (1 + rng.uniform(-magnitude, magnitude, values.shape)), 0.)
    if preserve_sum:
        observed_sum = np.where(valid, values, 0.).sum(axis=1)
        perturbed_sum = candidate.sum(axis=1)
        scale = np.divide(observed_sum, perturbed_sum, out=np.ones(len(values)), where=perturbed_sum > 0)
        candidate *= scale[:, None]
    rejected = valid & ((candidate < 0) | (candidate > upper) | ~np.isfinite(candidate))
    # Never clip or silently change an oxide sum to fix a rejected perturbation.
    if preserve_sum:
        rejected = valid & rejected.any(axis=1)[:, None]
    changed = valid & ~rejected
    output = np.where(changed, candidate, values)
    if cols:
        frame[cols] = output
    return int(valid.sum()), int((changed & (output != values)).sum()), int(rejected.sum()), invalid, int(valid.any(axis=1).sum())


def _missingness(frame, magnitude, rng):
    groups = [(OXIDES,100.), (TRACE_ELEMENTS,1e6)]
    eligible = changed = invalid = 0
    available = np.zeros(len(frame),dtype=bool)
    for names,upper in groups:
        cols,values,valid,bad = _chemical(frame,names,upper)
        mask=valid & (rng.random(values.shape)<magnitude)
        if cols:
            frame[cols]=np.where(mask,np.nan,values)
        eligible+=int(valid.sum())
        changed+=int(mask.sum())
        invalid+=bad
        available |= valid.any(axis=1)
    return eligible,changed,0,invalid,int(available.sum())


def _spatial(frame,magnitude,rng):
    if not {"Latitude","Longitude"} <= set(frame):
        return 0,0,0,0,0
    lat=frame.Latitude.to_numpy(dtype=float,copy=True)
    lon=frame.Longitude.to_numpy(dtype=float,copy=True)
    valid=np.isfinite(lat)&np.isfinite(lon)&(np.abs(lat)<=90)&(np.abs(lon)<=180)
    invalid=(~np.isnan(lat)&(~np.isfinite(lat)|(np.abs(lat)>90))) | (~np.isnan(lon)&(~np.isfinite(lon)|(np.abs(lon)>180)))
    # Spherical destination: fixed synthetic distance and randomized bearing.
    phi,lam=np.deg2rad(lat[valid]),np.deg2rad(lon[valid])
    theta=rng.uniform(0,2*np.pi,len(phi))
    angular=magnitude/6371008.8
    new_phi=np.arcsin(np.clip(np.sin(phi)*np.cos(angular)+np.cos(phi)*np.sin(angular)*np.cos(theta),-1.,1.))
    new_lam=lam+np.arctan2(np.sin(theta)*np.sin(angular)*np.cos(phi),np.cos(angular)-np.sin(phi)*np.sin(new_phi))
    lat[valid]=np.rad2deg(new_phi)
    lon[valid]=(np.rad2deg(new_lam)+180)%360-180
    frame["Latitude"],frame["Longitude"]=lat,lon
    n=int(valid.sum())
    return 2*n,2*n,0,int(invalid.sum()),n


def apply_stress(x, schema, scenario, magnitude, seed):
    if scenario not in SCENARIOS or magnitude not in SCENARIOS[scenario]:
        raise ValueError("Stress scenario is not predeclared")
    frame=x.copy(deep=True)
    rng=np.random.default_rng(seed)
    if scenario=="oxide_relative":
        evidence=_relative(frame,OXIDES,100.,magnitude,rng,preserve_sum=True)
    elif scenario=="trace_relative":
        evidence=_relative(frame,TRACE_ELEMENTS,1e6,magnitude,rng,preserve_sum=False)
    elif scenario=="missingness":
        evidence=_missingness(frame,magnitude,rng)
    else:
        evidence=_spatial(frame,magnitude,rng)
    if scenario in {"oxide_relative","missingness"} and any(name in frame for name in DERIVED):
        recomputed=add_geochemical_features(frame,version=schema.preparation_version)
        for col in DERIVED:
            if col in frame:
                frame[col]=recomputed[col]
    if not frame.columns.equals(x.columns) or not frame.index.equals(x.index):
        raise ValueError("Stress changed predictor identity")
    return StressInput(frame,*evidence)


def evaluate_stress(estimator, x_test, y_test, original_predictions, schema, *, fold, seed):
    base=regression_metrics(y_test,original_predictions)
    rows=[]
    for scenario,magnitudes in SCENARIOS.items():
        for magnitude in magnitudes:
            for offset in SEED_OFFSETS:
                trial_seed=seed+10000+fold*100+offset
                row={"fold":fold,"policy":POLICY,"scenario":scenario,"magnitude":magnitude,
                     "seed":trial_seed,"n_test":len(x_test),"status":"unavailable"}
                try:
                    stressed=apply_stress(x_test,schema,scenario,magnitude,trial_seed)
                    row.update({k:v for k,v in vars(stressed).items() if k!="frame"})
                    if stressed.eligible_cells:
                        pred=np.asarray(estimator.predict(stressed.frame),dtype=float)
                        metrics=regression_metrics(y_test,pred)
                        row.update({"status":"ok","rmse_ppm":metrics.rmse,"mae_ppm":metrics.mae,
                                    "rmse_delta_ppm":metrics.rmse-base.rmse,
                                    "mae_delta_ppm":metrics.mae-base.mae,
                                    "mean_prediction_shift_ppm":float(np.mean(np.abs(pred-original_predictions)))})
                    else:
                        row["reason"]="No eligible raw predictors"
                except (ValueError,TypeError,RuntimeError,OverflowError) as exc:
                    row.update(status="failed",reason=str(exc))
                rows.append(row)
    return rows
