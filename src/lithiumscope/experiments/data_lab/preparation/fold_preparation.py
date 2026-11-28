"""Freeze independent feature variants on all outer and inner TRAIN partitions."""
import numpy as np
import pandas as pd

from lithiumscope.core.reproducibility import canonical_json_hash
from lithiumscope.results.eda_protocol import fold_protocol
from lithiumscope.experiments.generalization.partitions import partitions
from .variants import VARIANTS, fit_variant, transform

BANDS = [-np.inf,0,5,10,20,40,60,80,100,126.12,696,np.inf]


def matrix_digest(frame):
    return canonical_json_hash({"ids":frame.index.astype(str).tolist(),
        "columns":list(frame.columns),
        "row_hashes":pd.util.hash_pandas_object(frame,index=False).astype(str).tolist()})


def prepare_folds(data, cfg, expected_protocol):
    x,y,groups,sources,eligible,numeric,categorical = data
    plan = partitions("model_1",data,cfg)
    protocol = fold_protocol(x,y,groups,[(a,b) for a,b,_ in plan],
                             eligible=eligible,strategy="group_kfold_spatial")
    if protocol != expected_protocol:
        raise ValueError("Prepared partitions differ from approved EDA protocol")
    rows, states, splits, coverage = [], [], [], []
    for outer,(train,test,inner) in enumerate(plan,1):
        splits.append({"fold":outer,"train":train.tolist(),"test":test.tolist(),
                       "inner":[{"train":a.tolist(),"validation":b.tolist()} for a,b in inner]})
        for i,(fit,reserved) in enumerate([(train,test),*inner]):
            fit = fit[eligible[fit]]
            xx, held = x.iloc[fit], x.iloc[reserved]
            if set(xx.index)&set(held.index) or set(groups.iloc[fit])&set(groups.iloc[reserved]):
                raise ValueError("Reserved membership entered TRAIN")
            if i and set(fit)&set(test):
                raise ValueError("Outer TEST entered inner TRAIN")
            band = pd.cut(y.iloc[fit],BANDS).astype("string")
            summary = pd.DataFrame({"band":band,"group":groups.iloc[fit],"source":sources.iloc[fit]})
            grouped = summary.groupby(["source","band"],dropna=False).agg(
                rows=("band","size"),groups=("group","nunique")).reset_index()
            grouped["outer_fold"],grouped["inner_fold"] = outer,i
            coverage.extend(grouped.to_dict(orient="records"))
            for arm in VARIANTS:
                context={"outer_fold":outer,"inner_fold":i,"arm":arm}
                try:
                    state=fit_variant(xx,numeric,categorical,arm)
                    a,b=transform(xx,state),transform(held,state)
                    if not a.index.equals(xx.index) or not b.index.equals(held.index) or list(a.columns)!=list(b.columns):
                        raise ValueError("Variant changed row identity or output schema")
                    if arm=="ilr_oxides" and not np.isfinite(a.filter(regex=r"^ilr_").to_numpy(float)).all():
                        raise ValueError("Nonfinite TRAIN ILR coordinates")
                    # Check identical inference for the first reserved row alone.
                    if len(held) and not transform(held.iloc[:1],state).equals(b.iloc[:1]):
                        # Floating-point BLAS may use a different order for a one-row matrix.
                        if not np.allclose(transform(held.iloc[:1],state)[state["numeric"]].to_numpy(float),
                                           b.iloc[:1][state["numeric"]].to_numpy(float),
                                           rtol=1e-12,atol=1e-12,equal_nan=True):
                            raise ValueError("Batch versus single-row inference differs")
                    row={**context,"state":"ready","fit_before":len(xx),"fit_after":len(a),
                         "reserved_rows":len(b),"features_before":len(x.columns),
                         "features_after":len(a.columns),"fit_sha256":matrix_digest(a),
                         "reserved_sha256":matrix_digest(b),"transform_sha256":canonical_json_hash(state),
                         "generated_rows":0,"removed_rows":0,"model_fits":0}
                    states.append({**context,"contract":state})
                except (ValueError,TypeError,KeyError,FloatingPointError) as exc:
                    row={**context,"state":"blocked","reason":str(exc),"model_fits":0}
                rows.append(row)
        print(f"    Fold {outer}/{len(plan)}: {1+len(inner)} particiones TRAIN revisadas",flush=True)
    return plan,protocol,pd.DataFrame(rows),states,splits,pd.DataFrame(coverage)
