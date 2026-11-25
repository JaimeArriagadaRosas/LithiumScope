"""M1 non-destructive source-scoped conflict and replicate audit."""
from __future__ import annotations

import pandas as pd

from lithiumscope.datasets.harmonization.observations import reconcile_observations


def audit_prepared_observations(prepared):
    frame = prepared.frame.copy().reset_index(drop=True)
    sample = next((c for c in ("source_sample", "Sample", "sample_id") if c in frame), None)
    replicate = next((c for c in ("replicate_id", "analysis_id") if c in frame), None)
    frame["_identity_source"] = frame["source_dataset"] if "source_dataset" in frame else ""
    frame["_identity_sample"] = frame[sample] if sample else ""
    frame["_identity_replicate"] = frame[replicate] if replicate else ""
    provenance = ["source_li_unit", "source_li_method", "source_publication", "source_material",
                  "Fe_total_provenance", "source_rock_class"]
    payload = list(dict.fromkeys([*prepared.schema.numeric, *prepared.schema.categorical,
                                  prepared.target, *[c for c in ("Longitude", "Latitude", *provenance) if c in frame]]))
    audit = reconcile_observations(frame, source="_identity_source", sample="_identity_sample",
                                  replicate="_identity_replicate", payload=payload)
    audit["row_index"] = prepared.frame.index.astype(str)
    for col in provenance:
        audit[col] = frame[col] if col in frame else ""
    # Same coordinates across distinct sources are a review candidate, not a proven duplicate.
    audit["cross_source_coordinate_match"] = False
    if {"Latitude", "Longitude"} <= set(frame):
        coords = frame[["Latitude", "Longitude"]].apply(pd.to_numeric, errors="coerce")
        candidates = coords.notna().all(axis=1) & frame["_identity_source"].fillna("").astype(str).str.strip().ne("")
        known = frame.loc[candidates, ["_identity_source"]].join(coords.loc[candidates])
        matched = known.groupby(["Latitude", "Longitude"])["_identity_source"].transform("nunique") > 1
        audit.loc[matched.index, "cross_source_coordinate_match"] = matched
    audit["training_cohort_mutated"] = False
    summary = {"rows": len(audit), "review_conflicts": int(audit.decision.eq("quarantine").sum()),
               "duplicate_candidates": int(audit.decision.eq("exclude").sum()),
               "unresolved_identity": int(audit.reason.eq("unresolved_identity_retained").sum()),
               "cross_source_coordinate_matches": int(audit.cross_source_coordinate_match.sum()),
               "cohort_mutated": False}
    return audit, summary
