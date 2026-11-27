"""Predeclared prediction-level transfer limits and inner-only policy selection."""
from __future__ import annotations

import numpy as np
import pandas as pd

BASE="B_balanced"
EXPERTS=("H_weight_010","I_weight_025")
POLICIES={BASE:(None,0.,None)}
for expert in EXPERTS:
    for fraction in (.25,.5,.75,1.):
        POLICIES[f"{expert}_blend_{int(fraction*100)}"]=(expert,fraction,None)
    for limit in (.5,1.,2.):
        POLICIES[f"{expert}_cap_{limit:g}"]=(expert,1.,limit)


def combine(base, expert, policy, train_scale):
    """Limit prediction disagreement using a scale from historical TRAIN.

    No evaluation labels or locations are consulted when applying a policy.
    """
    name,fraction,cap=POLICIES[policy]
    base=np.asarray(base,dtype=float)
    if base.ndim!=1 or not np.isfinite(base).all() or (base<0).any():
        raise ValueError("Invalid baseline predictions")
    if name is None:
        return base.copy()
    other=np.asarray(expert,dtype=float)
    if other.shape!=base.shape or not np.isfinite(other).all() or (other<0).any():
        raise ValueError("Invalid transfer predictions")
    if not np.isfinite(train_scale) or train_scale<0:
        raise ValueError("Invalid historical TRAIN scale")
    delta=other-base
    if cap is not None:
        limit=cap*np.maximum(base,max(1.,train_scale))
        delta=np.clip(delta,-limit,limit)
    return np.maximum(base+fraction*delta,0.)


def _rmse(errors):
    return float(np.sqrt(np.mean(np.asarray(errors,float)**2)))


def select_inner(table):
    """Pool inner OOF only. Guard MAE, mean fold RMSE, low and mid bands."""
    required={"position","inner_fold","truth","policy","prediction"}
    if not required.issubset(table.columns) or set(table.policy)!=set(POLICIES):
        raise ValueError("Incomplete inner policy evidence")
    control=table.loc[table.policy.eq(BASE)].sort_values("position").reset_index(drop=True)
    if not len(control) or control.position.duplicated().any():
        raise ValueError("Invalid inner control identities")
    keys=["position","inner_fold","truth"]
    low=control.truth.le(100).to_numpy()
    mid=(control.truth.gt(100)&control.truth.le(696)).to_numpy()
    high=control.truth.gt(696).to_numpy()
    results=[]
    for name,part in table.groupby("policy",sort=False):
        part=part.sort_values("position").reset_index(drop=True)
        if not part[keys].equals(control[keys]):
            raise ValueError("Inner evaluation identities differ")
        err=(part.prediction-part.truth).to_numpy(float)
        if not np.isfinite(err).all():
            raise ValueError("Nonfinite inner predictions")
        results.append({"policy":name,"rmse":_rmse(err),"mae":float(np.abs(err).mean()),
            "mean_fold_rmse":float(np.mean([_rmse(err[part.inner_fold.eq(f).to_numpy()])
                                            for f in part.inner_fold.unique()])),
            "low_rmse":_rmse(err[low]) if low.any() else None,
            "mid_rmse":_rmse(err[mid]) if mid.any() else None,
            "high_rmse":_rmse(err[high]) if high.any() else None,
            "low_rows":int(low.sum()),"mid_rows":int(mid.sum()),"high_rows":int(high.sum())})
    baseline=next(r for r in results if r["policy"]==BASE)
    for row in results:
        row["guard_pass"]=row["policy"]==BASE or (low.sum()>=5 and mid.sum()>=5
            and all(row[k]<=baseline[k]+1e-10 for k in
                    ("rmse","mae","mean_fold_rmse","low_rmse","mid_rmse")))
    choice=min((r for r in results if r["guard_pass"]),
               key=lambda r:(r["rmse"],r["mae"],r["policy"]!=BASE,r["policy"]))
    return choice["policy"],results
