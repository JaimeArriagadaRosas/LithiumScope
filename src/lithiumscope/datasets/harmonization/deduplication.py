"""Conservative, auditable detection of duplicates in harmonized sources."""
from __future__ import annotations

import numpy as np
import pandas as pd

IDENTIFIERS = ("source_sample", "sample_id", "Sample")
COORDINATES = ("Longitude", "Latitude", "Longitude (X)", "Latitude (Y)", "Logintude (X)", "longitude", "latitude")
PROVENANCE = {"source_dataset", "source_file", "source_row", "source_path", "source_sha256"}


def duplicate_evidence(frame: pd.DataFrame, *, coordinate_decimals: int) -> tuple[pd.Series, dict]:
    if not 0 <= coordinate_decimals <= 12:
        raise ValueError("coordinate_decimals must be between 0 and 12")
    identifiers = [name for name in IDENTIFIERS if name in frame]
    payload = [name for name in frame if name not in set(IDENTIFIERS) | set(COORDINATES) | PROVENANCE]
    keep = pd.Series(True, index=frame.index, dtype=bool)
    groups: dict[tuple, list[int]] = {}
    unresolved = []
    events = []
    for index, row in frame.iterrows():
        ids = {str(row[name]).strip() for name in identifiers if pd.notna(row[name]) and str(row[name]).strip()}
        try:
            lon, lat, li = (float(row[name]) for name in ("Longitude", "Latitude", "Li_icpms"))
        except (KeyError, TypeError, ValueError):
            unresolved.append(int(index))
            continue
        if len(ids) != 1 or not all(np.isfinite([lon, lat, li])) or not (-180 <= lon <= 180 and -90 <= lat <= 90 and li >= 0):
            unresolved.append(int(index))
            continue
        key = (next(iter(ids)), round(lon, coordinate_decimals), round(lat, coordinate_decimals))
        groups.setdefault(key, []).append(int(index))
    for indices in groups.values():
        if len(indices) < 2:
            continue
        # Require equal exact coordinates and all substantive measurements, including missingness.
        comparison = frame.loc[indices, ["Longitude", "Latitude", *payload]]
        if len(comparison.drop_duplicates()) != 1:
            events.append({"state": "conflict_retained", "rows": indices})
            continue
        for index in indices[1:]:
            keep.loc[index] = False
            events.append({"state": "identical_measurements", "kept_row": indices[0], "removed_row": index})
    return keep, {
        "rows_before_deduplication": len(frame),
        "rows_after_deduplication": int(keep.sum()),
        "duplicates_removed": int((~keep).sum()),
        "deduplication_columns": [*identifiers, "Longitude", "Latitude", *payload],
        "coordinate_decimals": coordinate_decimals,
        "identifier_candidates": identifiers,
        "unresolved_rows_retained": unresolved,
        "events": events,
        "rule": "complete identity, exact coordinates, identical substantive measurements; retain conflicts",
    }
