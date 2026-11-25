"""Trace source lots without using identifiers as predictors or changing targets."""
from __future__ import annotations

import pandas as pd


def summarize_batches(source, prepared):
    raw = pd.read_csv(source, low_memory=False)
    if "Sample_material_batch_id" not in raw:
        return {"state":"unavailable","reason":"No source batch identifier","batches":[]}
    positions = pd.to_numeric(pd.Series(prepared["row_ids"]), errors="raise").to_numpy(int)
    if len(set(positions)) != len(positions) or min(positions) < 0 or max(positions) >= len(raw):
        raise ValueError("Prepared M1 row identifiers do not map uniquely to source")
    accepted = raw.iloc[positions].copy()
    target = pd.Series(prepared["target"],index=accepted.index,dtype=float)
    original = pd.to_numeric(accepted["Li_icpms"],errors="coerce")
    # Censored source strings can be valid under the canonical parser; do
    # not certify raw textual values as numeric measurements.
    comparable = original.notna()
    if not ((original[comparable]-target[comparable]).abs() <= 1e-9).all():
        raise ValueError("Prepared M1 target differs from numeric source")
    accepted["__target"] = target
    accepted["__group"] = prepared["groups"]
    key = "Sample_material_batch_id"
    batches = []
    for batch, frame in accepted.loc[accepted[key].notna()].groupby(key):
        values=frame["__target"]
        batches.append({"batch":str(batch),"raw_rows":int(raw[key].eq(batch).sum()),
            "prepared_rows":len(frame),"groups":int(frame["__group"].nunique()),
            "minimum":float(values.min()),"median":float(values.median()),"maximum":float(values.max()),
            "row_ids":[str(i) for i in frame.index]})
    batches.sort(key=lambda b:b["maximum"],reverse=True)
    return {"state":"complete","numeric_target_rows_verified":int(comparable.sum()),
            "unassigned_rows":int(accepted[key].isna().sum()),"batches":batches}
