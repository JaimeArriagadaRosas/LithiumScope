"""Read a completed local v5 audit without changing the canonical representation."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.paths import PROJECT_ROOT


def load_physical_comparison(directory, source, frame, v4, mask):
    directory=Path(directory).resolve()
    if not directory.is_relative_to(PROJECT_ROOT.resolve()):
        raise ValueError("Physical audit must belong to this workspace")
    report=json.loads((directory/"summary.json").read_text(encoding="utf-8"))
    if (report.get("source_sha256",{}).get("manifest")!=file_sha256(source)
            or report.get("samples")!=len(frame) or report.get("verified_extracted")!=len(frame)
            or report.get("rows_with_all_finite_descriptors")!=len(frame)):
        raise ValueError("Physical audit is not aligned or complete")
    names=("features_v5_164.csv","sample_audit.csv")
    for name in names:
        path=(directory/name).resolve()
        if not path.is_relative_to(directory) or report.get("artifacts",{}).get(name)!=file_sha256(path):
            raise ValueError("Physical audit artifact hash mismatch")
    physical=pd.read_csv(directory/names[0])
    evidence=pd.read_csv(directory/names[1])
    manifest=pd.read_csv(source)
    expected=manifest.sample_id.astype(str).tolist()
    if (physical.sample_id.astype(str).tolist()!=expected or evidence.sample_id.astype(str).tolist()!=expected
            or not np.array_equal(evidence.Li_icpms,manifest.Li_icpms)
            or not np.array_equal(evidence.spatial_group.astype(str),manifest.spatial_group.astype(str))
            or not evidence.state.eq("extracted").all()):
        raise ValueError("Physical rows, labels or groups changed")
    columns=[c for c in v4 if not c.startswith("nir_swir22_nd_")]
    if list(physical.columns[1:])!=columns or not np.isfinite(physical[columns].to_numpy(float)).all():
        raise ValueError("Physical feature schema differs")
    valid=evidence.eligible_for_experimental_fit.to_numpy()
    if valid.dtype.kind!="b":
        raise ValueError("Expected explicit boolean physical eligibility")
    keep=np.asarray(mask,dtype=bool)&valid
    if not keep.any():
        raise ValueError("No common eligible TRAIN rows")
    return physical[columns].set_axis(v4.index),keep
