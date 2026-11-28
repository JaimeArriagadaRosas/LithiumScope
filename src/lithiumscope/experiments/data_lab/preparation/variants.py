"""Serializable TRAIN-fitted feature contracts for the final M1 preflight.

No target, source ID, sample ID or validation labels enter a transformation.
The same transform() is the future inference contract, with no refitting.
"""
import numpy as np
import pandas as pd

from lithiumscope.model_1.training.preprocessing.tabular import OXIDES, TRACE_ELEMENTS

VARIANTS = ("reference", "without_categories", "ilr_oxides", "chemistry_missing_indicators")
CHEMISTRY = OXIDES + TRACE_ELEMENTS


def helmert(size):
    """Fixed orthonormal basis of the zero-sum log-composition subspace."""
    basis = np.zeros((size, size-1), dtype=float)
    for j in range(1,size):
        basis[:j,j-1] = 1 / np.sqrt(j*(j+1))
        basis[j,j-1] = -j / np.sqrt(j*(j+1))
    return basis


def fit_variant(train, numeric, categorical, variant):
    if variant not in VARIANTS or list(train.columns) != [*numeric,*categorical] or not len(train):
        raise ValueError("Invalid variant or TRAIN schema")
    state = {"variant":variant,"input_columns":list(train.columns),
             "numeric":list(numeric),"categorical":list(categorical),
             "fit_rows":len(train),"fit_scope":"TRAIN_only","target_used":False}
    if variant == "without_categories":
        state["categorical"] = []
    elif variant == "ilr_oxides":
        if not set(OXIDES) <= set(numeric):
            raise ValueError("ILR requires all ten oxide columns")
        ox = train[OXIDES].apply(pd.to_numeric, errors="raise")
        if (ox.lt(0) | np.isinf(ox)).any().any():
            raise ValueError("Negative or infinite oxide values require analytical review")
        positive = ox.where(ox.gt(0))
        medians, minima = positive.median(),positive.min()
        if medians.isna().any() or minima.isna().any():
            raise ValueError("No positive TRAIN measurements for an oxide")
        # Explicit exploratory policy: missing uses TRAIN positive median;
        # reported zero uses half TRAIN smallest positive, not a certified LOD.
        state["missing_fill"] = medians.to_dict()
        state["zero_fill"] = (minima*.5).to_dict()
        state["zero_policy"] = "half_min_positive_TRAIN_not_certified_detection_limit"
        state["missing_policy"] = "positive_TRAIN_median_experimental"
        state["basis"] = helmert(len(OXIDES)).tolist()
        state["components"] = list(OXIDES)
        state["numeric"] = [c for c in numeric if c not in OXIDES]+[f"ilr_{i}" for i in range(1,len(OXIDES))]
    elif variant == "chemistry_missing_indicators":
        observed = train[[c for c in CHEMISTRY if c in numeric]].isna()
        state["indicator_columns"] = [c for c in observed if observed[c].any() and not observed[c].all()]
        if not state["indicator_columns"]:
            raise ValueError("No variable chemistry missingness in TRAIN")
        state["numeric"] += ["missing_"+c for c in state["indicator_columns"]]
    return state


def transform(frame, state):
    if list(frame.columns) != state["input_columns"]:
        raise ValueError("Variant input schema changed")
    variant = state["variant"]
    result = frame.copy()
    if variant == "ilr_oxides":
        components = state["components"]
        ox = result[components].apply(pd.to_numeric,errors="raise")
        if (ox.lt(0) | np.isinf(ox)).any().any():
            raise ValueError("Negative or infinite oxide values require analytical review")
        for c in components:
            ox[c] = ox[c].fillna(state["missing_fill"][c]).mask(ox[c].eq(0),state["zero_fill"][c])
        if not np.isfinite(ox).all().all() or not ox.gt(0).all().all():
            raise ValueError("Invalid completed composition")
        # The Helmert contrasts annihilate the common log(total); closure
        # is implicit. Never use the imputed values for QC or physical sums.
        projected = np.log(ox.to_numpy(float)) @ np.asarray(state["basis"],float)
        for j in range(projected.shape[1]):
            result[f"ilr_{j+1}"] = projected[:,j]
    elif variant == "chemistry_missing_indicators":
        for c in state["indicator_columns"]:
            result["missing_"+c] = result[c].isna().astype(float)
    return result.loc[:,state["numeric"]+state["categorical"]]
