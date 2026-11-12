"""Train-only optical coverage policy for the full six-algorithm M2 competition.

The 783-row source manifest, expanded 77 GEOROC observations, all outer
held-out examples, five anchored folds, model families and optimizer stay
unchanged. Only fitting subsets can exclude pixels with too little six-band
coverage. No image is rewritten and no sample is silently removed from OOF.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.paths import PROJECT_ROOT


def training_quality_mask(
    manifest_path: Path,
    *,
    inventory_path: Path,
    minimum_joint_valid: float=0.95,
) -> tuple[np.ndarray,dict]:
    """Validate frozen inventory alignment and return eligible fitting rows."""
    if not (0 < minimum_joint_valid <= 1):
        raise ValueError("Quality minimum must be in (0,1]")
    manifest=pd.read_csv(manifest_path)
    inventory=pd.read_csv(inventory_path)
    if len(manifest)!=len(inventory) or not len(manifest):
        raise ValueError("Quality inventory count must equal complete training manifest")
    if not {"sample_id","image_path"}.issubset(manifest):
        raise ValueError("Model 2 manifest lacks sample identity or image path")
    if not {"sample_id","image_sha256","sixband_joint_valid_fraction",
            "manifest_index"}.issubset(inventory):
        raise ValueError("Scientific inventory is missing coverage evidence")
    if inventory.manifest_index.astype(int).tolist()!=list(range(len(manifest))):
        raise ValueError("Scientific inventory indices differ from manifest")
    if not np.array_equal(
        manifest.sample_id.astype(str).to_numpy(),
        inventory.sample_id.astype(str).to_numpy(),
    ):
        raise ValueError("Scientific inventory sample order differs from manifest")
    valid=inventory.sixband_joint_valid_fraction.to_numpy(dtype=float)
    if not np.isfinite(valid).all() or ((valid<0)|(valid>1)).any():
        raise ValueError("Scientific inventory includes invalid coverage fractions")
    # Replay immutable source hashes. Multiple labels can reference one NPY.
    checked={}
    for path,digest in zip(manifest.image_path,inventory.image_sha256):
        p=Path(path)
        if p not in checked:
            if not p.is_file():raise FileNotFoundError(p)
            checked[p]=file_sha256(p)
        if checked[p]!=str(digest):
            raise ValueError(f"Original spectral image changed since QA: {p}")
    keep=np.asarray(valid>=minimum_joint_valid,dtype=bool)
    if not keep.any():
        raise ValueError("No training data eligible after quality filter")
    summary={
        "policy":"train_only_sixband_joint_valid",
        "minimum_joint_valid":float(minimum_joint_valid),
        "manifest_rows":int(len(manifest)),
        "training_eligible":int(keep.sum()),
        "training_excluded":int((~keep).sum()),
        "all_test_rows_preserved":True,
        "images_hashed":len(checked),
        "quality_inventory":str(inventory_path),
        "manifest_path":str(manifest_path),
        "field_rock_certification":False,
    }
    return keep,summary
