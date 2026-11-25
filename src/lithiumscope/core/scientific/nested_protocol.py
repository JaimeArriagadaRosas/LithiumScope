"""Audit nested group partitions before any estimator fit.

The same deterministic GroupKFold used by canonical training is inspected here.
No coordinates, labels or metrics from outer TEST select an inner partition.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from lithiumscope.core.scientific.spatial_audit import audit_spatial_folds
from lithiumscope.core.scientific_checks import assert_partition_coverage, assert_group_isolation


def _audit_parent(frame, groups, target, parent, *, outer_fold, phase, folds,
                  minimum_distance_m, patch_side_m, endpoint, quantile):
    """Parent indexes refer to the original ordered cohort, never a held-out fold."""
    local_groups = groups.iloc[parent].reset_index(drop=True)
    if local_groups.nunique() < folds:
        return [{"outer_fold": outer_fold, "phase": phase, "state": "blocked",
                 "reason": "insufficient_groups", "parent_rows": len(parent)}], []
    local_frame = frame.iloc[parent].reset_index(drop=True)
    splits = list(GroupKFold(n_splits=folds).split(local_frame, groups=local_groups))
    spatial, _ = audit_spatial_folds(local_frame, local_groups, splits,
                                    minimum_distance_m=minimum_distance_m, patch_side_m=patch_side_m)
    result = []
    for i, (tr, va) in enumerate(splits):
        row = spatial.iloc[i].to_dict()
        row.update(outer_fold=outer_fold, phase=phase, parent_rows=len(parent),
                   state="ready" if row["distance_policy_pass"] else "blocked",
                   reason=None if row["distance_policy_pass"] else "minimum_distance")
        if endpoint is not None or quantile is not None:
            cut = float(np.quantile(target[parent[tr]], quantile)) if quantile is not None else float(endpoint)
            train_y, valid_y = target[parent[tr]] >= cut, target[parent[va]] >= cut
            row.update(label_threshold_ppm=cut, train_positive=int(train_y.sum()),
                       train_negative=int((~train_y).sum()), valid_positive=int(valid_y.sum()),
                       valid_negative=int((~valid_y).sum()))
            if not train_y.any() or train_y.all() or not valid_y.any() or valid_y.all():
                row.update(state="blocked", reason="single_class")
        result.append(row)
    return result, [(parent[tr], parent[va]) for tr, va in splits]


def audit_nested_development(frame, groups, target, outer_splits, *, inner_folds,
                             eligible=None, endpoints=None, final_endpoint=None,
                             quantile=.75, fixed_endpoint=None, minimum_distance_m=0.,
                             patch_side_m=None, optimization=True, operating=False,
                             calibration=False):
    """Audit Optuna, operating crossfit, nested calibration and final development.

    Endpoints are fixed at enclosing TRAIN for operating/calibration; an
    optimization split recomputes its endpoint from inner TRAIN unless fixed.
    The final development parent contains only QA-eligible rows.
    """
    n = len(frame)
    if type(inner_folds) is not int or inner_folds < 2:
        raise ValueError("Invalid inner fold count")
    if not isinstance(groups, pd.Series) or not frame.index.equals(groups.index):
        raise ValueError("Nested spatial groups are not aligned")
    if isinstance(target, pd.Series) and not frame.index.equals(target.index):
        raise ValueError("Nested target is not aligned")
    target = np.asarray(target, dtype=float)
    mask = np.ones(n, dtype=bool) if eligible is None else np.asarray(eligible)
    if target.shape != (n,) or not np.isfinite(target).all() or (target < 0).any():
        raise ValueError("Invalid nested lithium target")
    if mask.shape != (n,) or mask.dtype.kind != "b" or not mask.any():
        raise ValueError("Nested eligibility must be a nonempty aligned boolean mask")
    if groups.isna().any() or groups.astype(str).str.strip().eq("").any():
        raise ValueError("Missing nested spatial groups")
    outer_splits = list(outer_splits)
    assert_partition_coverage(outer_splits, n)
    classification = endpoints is not None
    if classification:
        if (len(endpoints) != len(outer_splits) or not np.isfinite(endpoints).all()
                or final_endpoint is None or not np.isfinite(final_endpoint)
                or not 0 < quantile < 1):
            raise ValueError("Invalid nested label endpoints")
    elif operating or calibration or final_endpoint is not None:
        raise ValueError("Classification endpoints required for operating/calibration")
    if fixed_endpoint is not None and (not np.isfinite(fixed_endpoint) or fixed_endpoint < 0):
        raise ValueError("Invalid fixed endpoint")
    parents = []
    for i, (tr, te) in enumerate(outer_splits, 1):
        assert_group_isolation(tr, te, groups)
        tr = np.asarray(tr)
        parents.append((i, tr[mask[tr]], None if endpoints is None else endpoints[i-1]))
    parents.append((0, np.flatnonzero(mask), final_endpoint))
    records = []
    for fold, parent, endpoint in parents:
        def audit(rows, phase, cut=None, q=None):
            result, splits = _audit_parent(
                frame, groups, target, rows, outer_fold=fold, phase=phase, folds=inner_folds,
                minimum_distance_m=minimum_distance_m, patch_side_m=patch_side_m,
                endpoint=cut, quantile=q)
            records.extend(result)
            return splits
        if optimization:
            audit(parent, "optimization", fixed_endpoint if classification else None,
                  quantile if classification and fixed_endpoint is None else None)
        if operating:
            crossfits = audit(parent, "operating", endpoint)
            if calibration:
                for i, (fit, _) in enumerate(crossfits, 1):
                    audit(fit, f"operating_calibration_{i}", endpoint)
        if calibration:
            audit(parent, "estimator_calibration", endpoint)
    return pd.DataFrame(records)


def nested_summary(table: pd.DataFrame) -> dict:
    if table.empty:
        return {"state": "not_requested", "partitions": 0, "ready": 0, "blocked": 0}
    blocked = int(table.state.eq("blocked").sum())
    return {"state": "blocked" if blocked else "ready", "partitions": len(table),
            "ready": int(table.state.eq("ready").sum()), "blocked": blocked,
            "phase_counts": {str(k): int(v) for k, v in table.phase.value_counts().items()},
            "fit_performed": False}
