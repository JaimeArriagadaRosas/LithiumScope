"""Predeclared nested whole-group learning curves; diagnostic only, never promotion."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from lithiumscope.core.scientific_checks import assert_group_isolation, assert_partition_coverage


@dataclass(frozen=True)
class CurvePartition:
    fold: int
    fraction: float
    seed: int
    train: np.ndarray
    test: np.ndarray
    train_groups: int
    test_groups: int
    endpoint_ppm: float | None
    status: str


def build_curve_plan(y, groups, splits, *, kind, eligible=None, endpoints=None,
                     fractions=(.25,.5,.75,1.), seeds=(42,43,44)):
    y=np.asarray(y,dtype=float)
    groups=np.asarray(groups)
    if y.ndim!=1 or not len(y) or not np.isfinite(y).all() or groups.shape!=y.shape:
        raise ValueError("Invalid learning curve cohort")
    if pd.isna(groups).any() or any(not str(g).strip() for g in groups):
        raise ValueError("Missing spatial groups")
    if kind not in {"regression","classification"}:
        raise ValueError("Invalid learning curve kind")
    fractions=tuple(fractions);seeds=tuple(seeds)
    if not fractions or tuple(sorted(set(fractions)))!=fractions or not all(0<f<=1 for f in fractions) or fractions[-1]!=1.:
        raise ValueError("Fractions must be increasing, unique and end in 1")
    if not seeds or len(set(seeds))!=len(seeds) or any(type(s) is not int or s<0 for s in seeds):
        raise ValueError("Invalid curve seeds")
    keep=np.ones(len(y),dtype=bool) if eligible is None else np.asarray(eligible)
    if keep.shape!=y.shape or keep.dtype.kind!="b":
        raise ValueError("Eligibility must be an aligned boolean vector")
    splits=list(splits)
    assert_partition_coverage(splits,len(y))
    if kind=="classification":
        if endpoints is None or len(endpoints)!=len(splits) or not np.isfinite(endpoints).all():
            raise ValueError("Classification requires frozen outer TRAIN endpoints")
    elif endpoints is not None:
        raise ValueError("Regression has no classification endpoint")
    result=[]
    for fold,(tr,te) in enumerate(splits,1):
        tr=np.asarray(tr);te=np.asarray(te)
        assert_group_isolation(tr,te,pd.Series(groups))
        eligible_train=tr[keep[tr]]
        names=np.unique(groups[eligible_train].astype(str))
        if not len(names):
            raise ValueError("No eligible training groups")
        cut=float(endpoints[fold-1]) if endpoints is not None else None
        labels=(y>=cut).astype(int) if cut is not None else y
        for seed in seeds:
            order=np.random.default_rng(np.random.SeedSequence([seed,fold])).permutation(names)
            for fraction in fractions:
                # Full TRAIN is identical across seeds; do not count it as repeated data.
                if fraction==1. and seed!=seeds[0]:
                    continue
                n=min(len(names),max(1,math.ceil(fraction*len(names))))
                selected=eligible_train[np.isin(groups[eligible_train].astype(str),order[:n])]
                status="ready"
                if kind=="classification" and (np.unique(labels[selected]).size<2 or np.unique(labels[te]).size<2):
                    status="unavailable_single_class"
                elif len(selected)<2:
                    status="unavailable_small_train"
                result.append(CurvePartition(fold,float(fraction),seed,selected.copy(),te.copy(),n,
                                             len(set(groups[te])),cut,status))
    return result


def curve_plan_records(plan):
    return [{"fold":p.fold,"fraction_of_eligible_groups":p.fraction,"seed":p.seed,
             "n_train":len(p.train),"n_test":len(p.test),"train_groups":p.train_groups,
             "test_groups":p.test_groups,"endpoint_ppm":p.endpoint_ppm,"status":p.status,
             "train_positions":p.train.tolist(),"test_positions":p.test.tolist()} for p in plan]


def evaluate_curve(x,y,groups,plan,*,kind,algorithm,fit,predict):
    """fit callback must construct a fresh model using frozen per-fold parameters.
    It receives TRAIN rows only, never TEST or an outer-fold model instance.
    """
    y=pd.Series(y,index=x.index) if not isinstance(y,pd.Series) else y
    groups=pd.Series(groups,index=x.index) if not isinstance(groups,pd.Series) else groups
    if not x.index.equals(y.index) or not x.index.equals(groups.index):
        raise ValueError("Curve inputs must align")
    records=[]
    from lithiumscope.core.scientific.regression_metrics import regression_metrics
    from lithiumscope.core.scientific.classification_metrics import classification_metrics
    for p in plan:
        row={"algorithm":algorithm,"fold":p.fold,"fraction_of_eligible_groups":p.fraction,"seed":p.seed,
             "n_train":len(p.train),"n_test":len(p.test),"train_groups":p.train_groups,
             "test_groups":p.test_groups,"endpoint_ppm":p.endpoint_ppm,"status":p.status}
        if p.status!="ready":
            records.append(row);continue
        assert_group_isolation(p.train,p.test,groups)
        target=(y>=p.endpoint_ppm).astype(int) if kind=="classification" else y
        a,b=x.iloc[p.train],x.iloc[p.test]
        ya,yb=target.iloc[p.train],target.iloc[p.test]
        try:
            estimator=fit(p.fold,a,ya,groups.iloc[p.train])
            pred_train,pred_test=predict(estimator,a),predict(estimator,b)
            metric=(lambda t,v: regression_metrics(t,v).to_dict()) if kind=="regression" else classification_metrics
            row.update({f"train_{k}":v for k,v in metric(ya,pred_train).items()})
            row.update({f"test_{k}":v for k,v in metric(yb,pred_test).items()})
            row["status"]="ok"
        except (ValueError,RuntimeError,TypeError,ImportError) as exc:
            row.update(status="failed",error=f"{type(exc).__name__}: {exc}")
        records.append(row)
    return pd.DataFrame(records)
