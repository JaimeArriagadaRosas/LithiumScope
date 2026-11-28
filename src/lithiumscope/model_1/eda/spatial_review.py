"""Identity flags and spatial sensitivity diagnostics without changing folds or rows."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from lithiumscope.core.pipeline_audit import stable_source_row_id
from lithiumscope.core.scientific.eda_statistics import source_series
from lithiumscope.core.scientific.spatial_audit import EARTH_RADIUS_M, geographic_coordinates, nearest_distance
from lithiumscope.core.scientific.chemical_support import chemical_support
from lithiumscope.model_1.training.preprocessing.tabular import OXIDES, TRACE_ELEMENTS

UNKNOWN = {"", "unknown", "nan", "none", "<na>", "not reported", "n/a"}


def _text(frame, *names):
    out = pd.Series("", index=frame.index, dtype="string")
    for name in names:
        if name in frame:
            value = frame[name].astype("string").fillna("").str.strip().str.casefold()
            out = out.mask(out.eq("") & ~value.isin(UNKNOWN), value)
    return out


def identity_review(prepared):
    frame = prepared.frame
    source = source_series(frame)
    name = _text(frame, "source_sample", "Sample", "sample_id")
    publication = _text(frame, "source_publication", "Geochemistry Reference")
    uid = _text(frame, "source_unique_id")
    xy = frame.reindex(columns=["Latitude", "Longitude"]).apply(pd.to_numeric, errors="coerce")
    valid = np.isfinite(xy).all(axis=1) & xy.Latitude.between(-90,90) & xy.Longitude.between(-180,180)
    site = xy.round(5)
    site["source"] = source
    row_ids = [stable_source_row_id(prepared.audit.dataset_sha256, i) for i in frame.index]
    pub_sample = publication + "|" + name
    result = pd.DataFrame({"row_index": frame.index.astype(str), "row_id": row_ids,
        "source": source, "sample_name": name, "publication": publication, "unique_id": uid,
        "shared_name": name.ne("") & name.duplicated(keep=False),
        "shared_unique_id": uid.ne("") & uid.duplicated(keep=False),
        "shared_publication_sample": publication.ne("") & name.ne("") & pub_sample.duplicated(keep=False),
        "shared_site_in_source": valid & site.duplicated(keep=False),
        "identity_certified": False}, index=frame.index)
    for axis in ("Latitude", "Longitude"):
        lower = pd.to_numeric(frame.get(axis+"_min", pd.Series(np.nan,index=frame.index)), errors="coerce")
        upper = pd.to_numeric(frame.get(axis+"_max", pd.Series(np.nan,index=frame.index)), errors="coerce")
        result[axis.lower()+"_span_deg"] = (upper-lower).where(upper.ge(lower))
        result[axis.lower()+"_bounds_inverted"] = upper.lt(lower)
    # Nominal near-neighbour flags use only chemistry and location, never target.
    chemistry = frame.reindex(columns=[c for c in OXIDES+TRACE_ELEMENTS if c != "Fe2O3"]).apply(
        pd.to_numeric, errors="coerce").to_numpy(float)
    positions = np.flatnonzero(valid.to_numpy())
    pairs = []
    if len(positions):
        coords = np.radians(xy.iloc[positions].to_numpy(float))
        tree = BallTree(coords, metric="haversine")
        for k, neighbours in enumerate(tree.query_radius(coords, r=100/EARTH_RADIUS_M)):
            i = positions[k]
            for j in positions[neighbours]:
                if j <= i:
                    continue
                a, b = chemistry[i], chemistry[j]
                shared = np.isfinite(a) & np.isfinite(b)
                agrees = shared.sum() >= 5 and bool(np.all(
                    np.abs(a[shared]-b[shared]) <= np.maximum(.001,.02*np.maximum(np.abs(a[shared]),np.abs(b[shared])))))
                same_name = bool(name.iloc[i]) and name.iloc[i] == name.iloc[j]
                if same_name or agrees:
                    pairs.append({"left_row_id": row_ids[i], "right_row_id": row_ids[j],
                        "left_position": int(i), "right_position": int(j),
                        "same_sample_name": same_name, "shared_chemistry": int(shared.sum()),
                        "chemistry_agreement": agrees, "nominal_radius_m": 100,
                        "target_used": False, "duplicate_certified": False})
    columns = ["left_row_id","right_row_id","left_position","right_position","same_sample_name",
               "shared_chemistry","chemistry_agreement","nominal_radius_m","target_used","duplicate_certified"]
    return result.reset_index(drop=True), pd.DataFrame(pairs, columns=columns)


def spatial_sensitivity(prepared, splits, identity, pairs):
    coords = geographic_coordinates(prepared.frame)
    y, groups, x = prepared.y, prepared.groups, prepared.x
    chemistry = [c for c in OXIDES+TRACE_ELEMENTS if c in x]
    per_fold, rows, buffers = [], [], []
    for fold, (train, test) in enumerate(splits, 1):
        train, test = np.asarray(train), np.asarray(test)
        result = chemical_support(x.iloc[test], x.iloc[train], chemistry)
        geo, nearest = nearest_distance(coords, train, test)
        row = result.reset_index(drop=True)
        row.insert(0, "fold", fold)
        row["row_id"] = identity.iloc[test].row_id.to_numpy()
        row["nearest_train_row_id"] = identity.iloc[nearest].row_id.to_numpy()
        row["nominal_geographic_distance_m"] = geo
        row["above_train_max"] = y.iloc[test].gt(y.iloc[train].max()).to_numpy()
        rows.append(row)
        # Exact text/key overlaps are flags, not certified duplicate identities.
        overlap = {}
        for name in ("sample_name", "publication", "unique_id"):
            known = set(identity.iloc[train][name]) - {""}
            overlap[name+"_shared_test_rows"] = int(identity.iloc[test][name].isin(known).sum())
        cross = ((pairs.left_position.isin(train)&pairs.right_position.isin(test)) |
                 (pairs.left_position.isin(test)&pairs.right_position.isin(train)))
        per_fold.append({"fold": fold, "test_rows": len(test), "train_rows": len(train),
            "train_above_40": int(y.iloc[train].gt(40).sum()),
            "train_above_696": int(y.iloc[train].gt(696).sum()),
            "test_above_40": int(y.iloc[test].gt(40).sum()),
            "test_above_train_max": int(row.above_train_max.sum()),
            "chemistry_supported_test_rows": int(row.chemistry_nn_supported.sum()),
            "near_identity_pair_flags_across_fold": int(cross.sum()), **overlap})
        distances, _ = nearest_distance(coords, test, train)
        for threshold in (0, 100, 1000, 10000):
            keep = train[distances >= threshold]
            buffers.append({"fold": fold, "nominal_buffer_m": threshold,
                "train_rows": len(train), "train_rows_retained": len(keep),
                "train_rows_removed": len(train)-len(keep),
                "train_groups_retained": int(groups.iloc[keep].nunique()),
                "train_above_40_retained": int(y.iloc[keep].gt(40).sum()),
                "train_above_696_retained": int(y.iloc[keep].gt(696).sum()),
                "test_rows_unchanged": len(test), "applied_to_training": False,
                "coordinate_accuracy_certified": False})
    return {"outer_chemical_support": pd.concat(rows, ignore_index=True),
            "outer_identity_support_summary": pd.DataFrame(per_fold),
            "spatial_buffer_sensitivity": pd.DataFrame(buffers)}
