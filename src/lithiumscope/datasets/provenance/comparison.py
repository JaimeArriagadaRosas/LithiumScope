"""Non-destructive comparisons between candidate dataset versions."""
from __future__ import annotations

import numpy as np
import pandas as pd


def compare_versions(before: pd.DataFrame, after: pd.DataFrame, *, identity: tuple[str, ...],
                     target: str = "Li_icpms") -> dict:
    if not identity or len(set(identity)) != len(identity):
        raise ValueError("Declare distinct persistent identity columns")
    def indexed(frame):
        if not frame.columns.is_unique or not set(identity) <= set(frame):
            raise ValueError("Missing or ambiguous identity schema")
        ids = frame[list(identity)]
        if ids.isna().any().any() or any(ids[c].astype(str).str.strip().eq("").any() for c in identity):
            raise ValueError("Unresolved observation identity")
        if ids.duplicated().any():
            raise ValueError("Duplicate observation identity")
        return frame.set_index(list(identity))
    left, right = indexed(before), indexed(after)
    common = left.index.intersection(right.index, sort=False)
    columns = left.columns.intersection(right.columns, sort=False)
    a, b = left.loc[common, columns], right.loc[common, columns]
    changed = ~(a.eq(b).fillna(False) | (a.isna() & b.isna()))
    result = {"rows_before": len(left), "rows_after": len(right),
              "added": len(right.index.difference(left.index)), "removed": len(left.index.difference(right.index)),
              "matched": len(common), "modified": int(changed.any(axis=1).sum()),
              "added_columns": sorted(map(str, right.columns.difference(left.columns))),
              "removed_columns": sorted(map(str, left.columns.difference(right.columns))),
              "changed_cells": {str(c): int(changed[c].sum()) for c in columns},
              "missing_before": {str(c): int(left[c].isna().sum()) for c in left},
              "missing_after": {str(c): int(right[c].isna().sum()) for c in right}}
    if target in columns:
        x, y = pd.to_numeric(a[target], errors="coerce"), pd.to_numeric(b[target], errors="coerce")
        delta = (y - x).where(np.isfinite(x) & np.isfinite(y)).dropna()
        result["target_delta"] = {"comparable": len(delta), "mean": float(delta.mean()) if len(delta) else None,
                                  "max_abs": float(delta.abs().max()) if len(delta) else None}
    for name, limit in (("Latitude", 90), ("Longitude", 180)):
        if name in columns:
            x, y = pd.to_numeric(a[name], errors="coerce"), pd.to_numeric(b[name], errors="coerce")
            delta = (y - x).abs().where(x.between(-limit, limit) & y.between(-limit, limit)).dropna()
            if name == "Longitude":
                delta = delta.where(delta.le(180), 360 - delta)
            result[name.lower() + "_delta_deg"] = {"comparable": len(delta),
                "max_abs": float(delta.max()) if len(delta) else None}
    return result
