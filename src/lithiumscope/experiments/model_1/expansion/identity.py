"""Conservative identity resolution: never silently drop or merge distinct samples."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from lithiumscope.experiments.data_lab.candidate_screening import CHEM, sample, screen, text

KEY = ["source_dataset", "source_file", "source_row_position"]
MEASUREMENTS = ["Latitude", "Longitude", "Li_icpms", "SUM (no water)", *CHEM]
IDENTITY = ["source_unique_id", "source_publication", "source_sample"]
CONTEXT = ["Sample_type", "Rock_type", "source_tectonic_setting", "source_li_unit",
           "source_li_raw", "source_li_method", "source_coordinate_crs", "Fe_total_provenance"]


def row_ids(frame):
    if frame[KEY].isna().any().any() or frame.duplicated(KEY).any():
        raise ValueError("Candidate source row identity missing or duplicated")
    return pd.Series([hashlib.sha256(json.dumps(list(map(str, row)), ensure_ascii=False).encode()).hexdigest()
                      for row in frame[KEY].itertuples(index=False, name=None)], index=frame.index)


def resolve(candidates, base, groups):
    """Collapse exact candidate copies with strong identity; retain all decisions.

    Approximate matches and conflicting analyses remain review-only. Any match
    against the frozen source is still excluded by the original screening.
    """
    x = candidates.reset_index(drop=True).copy()
    ids = row_ids(x)
    pub = text(x, "source_publication")
    names = sample(x)
    uid = text(x, "source_unique_id")
    strong = uid.ne("") | (pub.ne("") & names.ne(""))
    # Use exact complete record equality, including missingness, and a strong
    # identity. A coordinate/Li coincidence alone never authorizes collapsing.
    fields = sorted(set(x.columns) - set(KEY) - {"raw_source_sha256"})
    signatures = x.reindex(columns=fields).copy()
    signatures["__uid"] = uid
    signatures["__pub"] = pub
    signatures["__sample"] = names
    signature = signatures.astype("string").fillna("<missing>")
    canonical = ids.copy()
    order = sorted(range(len(x)), key=lambda i: (
        str(x.iloc[i]["source_dataset"]) != "georoc_andean_202609",
        str(x.iloc[i]["source_file"]), int(x.iloc[i]["source_row_position"]), ids.iloc[i]))
    seen = {}
    for i in order:
        if not strong.iloc[i]:
            continue
        key = tuple(signature.iloc[i])
        if key in seen:
            canonical.iloc[i] = seen[key]
        else:
            seen[key] = ids.iloc[i]
    keep = canonical.eq(ids)
    original = screen(x, base, groups)
    reduced = screen(x.loc[keep].reset_index(drop=True), base, groups)
    original["candidate_id"] = ids
    original["canonical_id"] = canonical
    original["exact_copy"] = ~keep
    original["original_screening_state"] = original.screening_state
    for column in ("screening_state", "duplicate_id", "duplicate_publication_sample",
                   "duplicate_coordinate_li", "possible_near_duplicate"):
        original.loc[keep, column] = reduced[column].to_numpy()
    original.loc[~keep, "screening_state"] = "exact_candidate_copy"
    return original


def describe_domain(candidate_features, base_features, numeric):
    """Descriptive marginal coverage only; no fitted selection from TEST."""
    b = base_features[numeric].apply(pd.to_numeric, errors="coerce")
    c = candidate_features[numeric].apply(pd.to_numeric, errors="coerce")
    low, high = b.quantile(.01), b.quantile(.99)
    supported = low.notna() & high.notna()
    available = c.notna() & supported
    outside = ((c.lt(low) | c.gt(high)) & available).sum(axis=1)
    count = available.sum(axis=1)
    return pd.DataFrame({"predictors_present":candidate_features.notna().sum(axis=1),
        "numeric_observed":count, "marginal_ood_fraction":outside.div(count.replace(0,np.nan))},
        index=candidate_features.index)


def multivariate_support(candidate_x, train_x, numeric):
    """Robust chemistry-neighbour distance with explicit missingness indicators.

    All location/target variables are excluded from the chemistry index. The
    reference scale and 95th percentile of leave-one-out nearest distances
    are derived only from the supplied reference partition.
    """
    from sklearn.neighbors import NearestNeighbors

    cols=[c for c in numeric if c in CHEM]
    b=train_x[cols].apply(pd.to_numeric,errors="coerce").replace([np.inf,-np.inf],np.nan)
    c=candidate_x[cols].apply(pd.to_numeric,errors="coerce").replace([np.inf,-np.inf],np.nan)
    median=b.median()
    spread=b.quantile(.75)-b.quantile(.25)
    valid=(median.notna() & spread.gt(0) & b.notna().mean().ge(.5))
    cols=valid.index[valid].tolist()
    if len(cols)<5 or len(b)<2:
        return pd.DataFrame({"chemistry_nn_distance":np.nan,"chemistry_nn_q95":np.nan,
                             "chemistry_nn_supported":False},index=c.index)
    def matrix(frame):
        values=frame[cols]
        scaled=values.sub(median[cols]).div(spread[cols]).fillna(0.)
        return np.concatenate([scaled.to_numpy(float),values.isna().to_numpy(float)],axis=1)
    reference=matrix(b)
    nn=NearestNeighbors(n_neighbors=2).fit(reference)
    baseline=nn.kneighbors(reference,return_distance=True)[0][:,1]
    cutoff=float(np.quantile(baseline,.95))
    distances=nn.kneighbors(matrix(c),n_neighbors=1,return_distance=True)[0][:,0] if len(c) else np.array([])
    observed=c[cols].notna().sum(axis=1)
    supported=(observed.ge(max(5,int(np.ceil(len(cols)*.5)))) & (distances<=cutoff))
    return pd.DataFrame({"chemistry_nn_distance":distances,"chemistry_nn_q95":cutoff,
                         "chemistry_nn_supported":supported},index=c.index)
