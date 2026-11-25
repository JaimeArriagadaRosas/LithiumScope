"""Equal effective mass per exact original spectral signature, TRAIN only."""
from __future__ import annotations

import numpy as np
import pandas as pd


def signature_ids(x):
    values = x.to_numpy(float)
    if not np.isfinite(values).all() or values.ndim != 2 or not len(values):
        raise ValueError("Exact signature analysis requires finite spectral rows")
    _, inverse = np.unique(values, axis=0, return_inverse=True)
    return inverse.reshape(-1)


def signature_weights(x):
    ids = signature_ids(x)
    counts = np.bincount(ids)
    return 1.0 / counts[ids]


def signature_summary(x, labels=None):
    ids = signature_ids(x)
    counts = np.bincount(ids)
    result = {"unique_signatures": int(len(counts)), "repeated_rows": int(counts[counts > 1].sum()),
              "max_signature_multiplicity": int(counts.max())}
    if labels is not None:
        labels = np.asarray(labels, int)
        if labels.shape != ids.shape or not np.isin(labels, [0, 1]).all():
            raise ValueError("Invalid binary signature labels")
        positive = np.bincount(ids, weights=labels, minlength=len(counts))
        conflict = (positive > 0) & (positive < counts)
        result.update(conflicting_signatures=int(conflict.sum()), conflicting_rows=int(counts[conflict].sum()),
                      unavoidable_errors_same_signature=int(np.minimum(positive, counts-positive).sum()))
    return result


def signature_training(x, transformed, y, groups, mode):
    """For soft targets use weighted binary representatives, not relabeling.

    Each signature contributes total mass one. Its positive/negative masses
    are empirical class fractions, equivalent for additive binary losses.
    Tree minimum-leaf counts may still differ after compression.
    """
    if not x.index.equals(y.index) or not x.index.equals(groups.index):
        raise ValueError("Signature TRAIN identity mismatch")
    ids = signature_ids(x)
    labels = y.to_numpy(int)
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Signature treatment requires both classes")
    weights = signature_weights(x)
    if mode == "signature_weighted":
        return transformed, labels, weights, [
            {"row":str(x.index[i]), "action":"signature_weight", "weight":float(w),
             "signature":int(ids[i]), "group":str(groups.iloc[i])}
            for i,w in enumerate(weights)]
    if mode != "signature_soft":
        raise ValueError("Unknown signature policy")
    keep, mass, trace = [], [], []
    for identity in np.unique(ids):
        members = np.flatnonzero(ids == identity)
        for label in (0,1):
            same = members[labels[members] == label]
            if not len(same):
                continue
            rep = int(same[0])
            keep.append(rep)
            mass.append(float(len(same)/len(members)))
            for i in same:
                trace.append({"row":str(x.index[i]), "parent_a":str(x.index[rep]),
                    "action":"signature_soft_member", "weight":float(weights[i]),
                    "signature":int(identity), "group":str(groups.iloc[i])})
    return transformed[keep], labels[keep], np.asarray(mass), trace


def signature_overlap(train, test):
    """Exact row equality, no rounding and no target use."""
    values = pd.concat([train, test], ignore_index=True)
    ids = signature_ids(values)
    return int(np.isin(ids[len(train):], ids[:len(train)]).sum())
