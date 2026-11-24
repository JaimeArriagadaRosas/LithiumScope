"""Freeze the exact development fit and verify a trusted newly serialized candidate."""
from __future__ import annotations

import numpy as np
import pandas as pd

from lithiumscope.core.checkpoint_identity import _data_identity
from lithiumscope.core.reproducibility import canonical_json_hash, _package_versions


def freeze_final_fit(*, x, y, groups, row_ids, dataset_sha256, training_signature,
                     params, policy, reserved_ids=()):
    if not isinstance(x,pd.DataFrame) or x.empty or not x.index.is_unique or not x.columns.is_unique:
        raise ValueError("Final fit requires a nonempty unique predictor matrix")
    ids=list(map(str,row_ids))
    reserved=set(map(str,reserved_ids))
    if len(ids)!=len(x) or len(set(ids))!=len(ids) or any(not i.strip() for i in ids):
        raise ValueError("Final fit observation identities differ")
    if set(ids)&reserved:
        raise ValueError("Reserved independent holdout intersects final fit")
    if groups is None or len(groups)!=len(x) or pd.isna(groups).any():
        raise ValueError("Final fit requires spatial group identity")
    if hasattr(groups,"index") and not x.index.equals(groups.index):
        raise ValueError("Final fit groups are not aligned")
    if hasattr(y,"index") and not x.index.equals(y.index):
        raise ValueError("Final fit target is not aligned")
    target=np.asarray(y,dtype=float)
    if target.shape!=(len(x),) or not np.isfinite(target).all():
        raise ValueError("Final fit target is invalid")
    if not isinstance(dataset_sha256,str) or len(dataset_sha256)!=64:
        raise ValueError("Final fit dataset SHA is required")
    if not training_signature:
        raise ValueError("Final fit training signature is required")
    manifest={"version":1,"scope":"development_final_fit_not_external_validation",
              "dataset_sha256":dataset_sha256,"training_signature":training_signature,
              "row_ids":ids,"n_rows":len(ids),"spatial_groups":sorted(set(map(str,groups))),
              "predictors":_data_identity(x),"target":_data_identity(y),"groups":_data_identity(groups),
              "params":params,"policy":policy,"reserved_holdout_ids":sorted(reserved),
              "external_holdout_reserved":bool(reserved),"packages":_package_versions()}
    return {"manifest":manifest,"identity_sha256":canonical_json_hash(manifest)}


def verify_frozen_fit(contract):
    if not isinstance(contract,dict) or canonical_json_hash(contract["manifest"])!=contract["identity_sha256"]:
        raise ValueError("Final fit contract identity differs")
    return contract["manifest"]


def verify_serialized_candidate(original,restored,probe,*,model_group):
    if model_group=="model_1":
        schema=original["schema"]
        columns=list(schema.numeric)+list(schema.categorical)
        if restored["schema"]!=schema:
            raise ValueError("Serialized regression schema differs")
        a=original["estimator"].predict(probe)
        b=restored["estimator"].predict(probe)
        shape=(len(probe),)
    elif model_group=="model_2":
        columns=list(original["features"])
        if restored["features"]!=columns:
            raise ValueError("Serialized spectral feature schema differs")
        for bundle in (original,restored):
            if not np.array_equal(bundle["estimator"].classes_,[0,1]):
                raise ValueError("Serialized classifier class order differs")
        a=original["estimator"].predict_proba(probe)
        b=restored["estimator"].predict_proba(probe)
        shape=(len(probe),2)
        for values in (a,b):
            if not np.isfinite(values).all() or not np.all((values>=0)&(values<=1)) or not np.allclose(values.sum(axis=1),1,atol=1e-10,rtol=0):
                raise ValueError("Serialized classifier scores invalid")
    else:
        raise ValueError("Unsupported model group")
    if list(probe.columns)!=columns or not len(probe):
        raise ValueError("Serialization probe schema differs")
    a,b=np.asarray(a,float),np.asarray(b,float)
    if a.shape!=shape or b.shape!=shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Serialization prediction shape or finiteness differs")
    if not np.allclose(a,b,rtol=1e-12,atol=1e-10):
        raise ValueError("Serialized candidate changed predictions")
    if original.get("final_fit_contract")!=restored.get("final_fit_contract"):
        raise ValueError("Serialized final fit contract differs")
    return {"scope":"development_serialization_probe_not_evaluation","rows":len(probe),
            "maximum_absolute_delta":float(np.max(np.abs(a-b))),
            "tolerance":{"rtol":1e-12,"atol":1e-10},"status":"passed"}
