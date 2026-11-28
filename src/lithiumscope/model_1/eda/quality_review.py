"""Review QC exclusions and analytical target status without changing preparation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from lithiumscope.core.pipeline_audit import stable_source_row_id
from lithiumscope.core.scientific.eda_statistics import source_series
from lithiumscope.datasets.adapters.georoc.schema import MAJOR_OXIDES


def quality_review(prepared, data_config):
    raw = prepared.raw_frame
    lower, upper = float(data_config["quality_min"]), float(data_config["quality_max"])
    positions = {str(idx): idx for idx in raw.index}
    oxides = raw.reindex(columns=MAJOR_OXIDES).apply(pd.to_numeric, errors="coerce")
    measured = np.isfinite(oxides) & oxides.ge(0)
    complete = measured.all(axis=1)
    sums = oxides.where(measured).sum(axis=1, min_count=len(MAJOR_OXIDES))
    source_labels = source_series(raw)
    censoring = {(str(v["row_index"]), v["column"]): v for v in prepared.audit.censoring}
    qc = []
    for item in prepared.audit.exclusions:
        if item["stage"] != "06_quality_control":
            continue
        idx = positions[item["row_index"]]
        original = raw.loc[idx]
        censored = any((str(idx), name) in censoring for name in MAJOR_OXIDES)
        numeric_complete = bool(complete.loc[idx]) and not censored
        diagnostic_sum = float(sums.loc[idx]) if numeric_complete else None
        in_range = diagnostic_sum is not None and lower <= diagnostic_sum <= upper
        missing_total = item["reason"] == "missing_or_unparseable"
        alternative = {}
        for name in data_config["quality_column_candidates"]:
            if name == item["column"] or name not in raw:
                continue
            val = pd.to_numeric(original[name], errors="coerce")
            alternative[name] = val if pd.notna(val) and np.isfinite(val) else None
        status = ("review_sum_in_range" if missing_total and in_range else
                  "review_sum_out_of_range" if missing_total and numeric_complete else
                  "insufficient_original_measurements" if missing_total else "recorded_qc_exclusion")
        missing_oxides = "|".join(name for name in MAJOR_OXIDES if not measured.loc[idx,name])
        record = {**item, "measured_oxide_count": int(measured.loc[idx].sum()),
                  "missing_or_invalid_oxides": missing_oxides or "<none>",
                  "reported_source_file": original.get("source_file", None),
                  "reported_source_row_position": original.get("source_row_position", None),
                  "reported_source_unique_id": original.get("source_unique_id", None),
                  "raw_target": original.get(prepared.target, None),
                  "original_partition_identity_verified": False,
                  "all_ten_numeric_nonnegative": bool(complete.loc[idx]),
                  "censored_oxide": censored, "diagnostic_sum": diagnostic_sum,
                  "diagnostic_sum_in_range": in_range,
                  "iron_provenance": original.get("Fe_total_provenance", None),
                  "source_feot_raw": original.get("source_feot_raw", None),
                  "source_fe2o3_raw": original.get("source_fe2o3_raw", None),
                  "water_basis_and_units_certified": False,
                  "status": status, "approved_for_training": False}
        for name in MAJOR_OXIDES:
            record["raw_" + name] = original.get(name, None)
        for name, value in alternative.items():
            record["alternative_" + name] = value
        qc.append(record)
    qc_columns = ["stage", "row_index", "source", "sample_id", "source_row_id",
                  "column", "original", "reason", "measured_oxide_count",
                  "missing_or_invalid_oxides", "reported_source_file", "reported_source_row_position",
                  "reported_source_unique_id", "raw_target", "original_partition_identity_verified",
                  "all_ten_numeric_nonnegative", "censored_oxide", "diagnostic_sum",
                  "diagnostic_sum_in_range", "iron_provenance", "source_feot_raw",
                  "source_fe2o3_raw", "water_basis_and_units_certified", "status",
                  "approved_for_training", *["raw_" + name for name in MAJOR_OXIDES]]
    qc_rows = pd.DataFrame(qc) if qc else pd.DataFrame(columns=qc_columns)
    qc_summary = (qc_rows.groupby(["source", "reason", "status"], dropna=False)
                  .size().rename("rows").reset_index())
    missing_patterns = (qc_rows.loc[qc_rows.reason.eq("missing_or_unparseable")]
        .groupby(["source", "measured_oxide_count", "missing_or_invalid_oxides"], dropna=False)
        .size().rename("rows").reset_index())
    # Distinguish exact observed labels from substituted/censored and invalid labels.
    target_rows = []
    for idx, value in raw[prepared.target].items():
        numeric = pd.to_numeric(value, errors="coerce")
        flag = censoring.get((str(idx), prepared.target))
        finite = bool(pd.notna(numeric) and np.isfinite(numeric))
        if flag:
            status = "censored_or_substituted"
        elif not finite:
            status = "missing_or_invalid"
        elif numeric == 0:
            status = "reported_zero"
        elif numeric < 0:
            status = "negative"
        else:
            status = "positive_numeric"
        target_rows.append({"row_index": str(idx),
            "source_row_id": stable_source_row_id(prepared.audit.dataset_sha256, idx),
            "source": str(source_labels.loc[idx]), "raw_target": value,
            "status": status, "censor_type": flag.get("censor_type") if flag else None,
            "retained": idx in prepared.frame.index})
    target = pd.DataFrame(target_rows)
    target_summary = target.groupby(["source", "status", "retained"], dropna=False).size().rename("rows").reset_index()
    return {"qc_recovery_review": qc_rows, "qc_recovery_summary": qc_summary,
            "qc_missing_oxide_patterns": missing_patterns,
            "target_analytical_review": target, "target_analytical_summary": target_summary}
