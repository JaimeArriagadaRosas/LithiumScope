"""TRAIN-only selection, preserving tails, geology, sources and spatial coverage."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd

REFERENCE_ROWS = 3420
OASIS_ROWS = 681
MIN_COHORT_ROWS = 3 * OASIS_ROWS


@dataclass(frozen=True)
class BalanceArm:
    name: str
    cohort_target: int | None = None
    regime: str | None = None
    correction: bool = False


ARMS = (BalanceArm("control"), *(BalanceArm(f"balanced_{n}",cohort_target=n)
        for n in (3000,2700,2400,MIN_COHORT_ROWS)),
        BalanceArm("experts_quantiles",regime="quantiles"),
        BalanceArm("experts_fixed",regime="fixed"))


def balanced_positions(x,y,groups,sources,cohort_target,seed):
    if not MIN_COHORT_ROWS <= cohort_target <= REFERENCE_ROWS:
        raise ValueError("Cohort-equivalent minimum violated")
    if not (x.index.equals(y.index) and x.index.equals(groups.index) and x.index.equals(sources.index)):
        raise ValueError("Misaligned TRAIN rows")
    target=max(1,int(np.ceil(len(x)*cohort_target/REFERENCE_ROWS)))
    rng=np.random.default_rng(seed)
    rock=x.get("Rock_type",pd.Series("unknown",index=x.index)).fillna("unknown").astype(str)
    bands=np.searchsorted([5.,20.,50.,100.,696.,1000.],y.to_numpy(float),side="right")
    strata=list(zip(sources.astype(str),groups.astype(str),rock,bands,strict=True))
    # Keep every real TRAIN sample >=100 ppm; never remove rare high values to improve scores.
    protected=set(np.flatnonzero(y.to_numpy(float)>=100.))
    by_cell={}
    for i,key in enumerate(strata):
        by_cell.setdefault(key,[]).append(i)
    for cell in by_cell.values():
        if not protected.intersection(cell):
            # Prefer predictor completeness; randomly break ties reproducibly.
            quality=x.iloc[cell].notna().sum(axis=1).to_numpy()
            protected.add(int(rng.choice(np.asarray(cell)[quality==quality.max()])))
    if len(protected)>target:
        raise ValueError(f"Cannot reduce to {target} without losing protected tails or strata ({len(protected)})")
    remaining=np.asarray(sorted(set(range(len(x)))-protected),dtype=int)
    need=target-len(protected)
    counts=Counter(strata[i] for i in remaining)
    if need:
        weight=np.asarray([1/np.sqrt(counts[strata[i]]) for i in remaining])
        protected.update(rng.choice(remaining,need,replace=False,p=weight/weight.sum()).tolist())
    keep=np.asarray(sorted(protected),dtype=int)
    if len(keep)!=target or set(strata[i] for i in keep)!=set(strata):
        raise ValueError("Real-row balance failed coverage contract")
    return keep
