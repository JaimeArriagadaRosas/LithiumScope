"""Stable coordinates and a non-destructive candidate decision ledger."""
from __future__ import annotations

import numpy as np
import pandas as pd

from lithiumscope.core.reproducibility import canonical_json_hash


def coordinate_plan(frame: pd.DataFrame, *, minimum_distance_km: float = 100,
                    decimals: int = 4) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"candidate_gate", "nearest_full_mamani_km", "latitude", "longitude"}
    if not required <= set(frame) or not frame.columns.is_unique:
        raise ValueError("Missing or ambiguous candidate fields")
    if not np.isfinite(minimum_distance_km) or minimum_distance_km < 0 or not isinstance(decimals, int) or not 0 <= decimals <= 8:
        raise ValueError("Invalid coordinate planning policy")
    reserved = {"image_id", "coordinate_key", "plan_state", "plan_reason", "plan_row"}
    if reserved & set(frame):
        raise ValueError("Candidate input already contains plan fields")
    ledger = frame.copy().reset_index(drop=True)
    ledger["plan_row"] = range(len(ledger))
    lat, lon, distance = (pd.to_numeric(ledger[k], errors="coerce") for k in
                          ("latitude", "longitude", "nearest_full_mamani_km"))
    valid = lat.between(-90, 90) & lon.between(-180, 180)
    far = np.isfinite(distance) & distance.ge(minimum_distance_km)
    eligible = ledger.candidate_gate.eq("REVIEW_METHOD_AND_SURFACE").fillna(False)
    ledger["plan_reason"] = np.select(
        [~valid, ~far, ~eligible], ["invalid_coordinates", "distance_missing_or_below_policy", "screening_gate"],
        default="selected_for_exploratory_imagery")
    selected = valid & far & eligible
    ledger["plan_state"] = np.where(selected, "selected", "not_selected")
    ledger["coordinate_key"] = ""
    ledger["image_id"] = ""
    rows = {}
    for i in ledger.index[selected]:
        # This new protocol explicitly uses the rounded location as patch center.
        y, x = round(float(lat[i]), decimals), round(float(lon[i]), decimals)
        y, x = y or 0.0, x or 0.0
        key = f"{y:.{decimals}f}|{x:.{decimals}f}"
        image_id = "coord_" + canonical_json_hash({"coordinate": key, "decimals": decimals})[:24]
        ledger.loc[i, ["coordinate_key", "image_id"]] = [key, image_id]
        if image_id in rows and rows[image_id]["coordinate_key"] != key:
            raise ValueError("Coordinate identity collision")
        rows[image_id] = {"image_id": image_id, "latitude": y, "longitude": x, "coordinate_key": key}
    sites = pd.DataFrame(list(rows.values()), columns=["image_id", "latitude", "longitude", "coordinate_key"])
    return sites.sort_values("image_id").reset_index(drop=True), ledger
