"""Local frozen v4 evidence indexed by image bytes, coordinates and scene, not user sample ID."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import shutil
from uuid import uuid4

import numpy as np
import pandas as pd

from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.paths import PROJECT_ROOT
from lithiumscope.tools.acquisition.sentinel2.rededge import cached_rededge_sample

MANIFEST_SHA = "5c42e8b5fac697d6c1a579ef760ad642d736da14fa4e7d605596b7f4813498e1"
QUALITY_SHA = "273d7bb4156bed35b3d88f1ce4b000173a7ffd5dfd1bc7c159718f52d21a7dcb"
V4_DIR=Path("results/scientific_simulations/m2_rededge_v4_20261113")
CATALOG=Path("data/processed/model_2/frozen_cohorts/v4_706")


def register_frozen_catalog(root=PROJECT_ROOT):
    """Copy known immutable CSV evidence; future changes to training manifest are independent."""
    root=Path(root).resolve()
    files={
        "training_manifest.csv":(root/"data/processed/model_2/training_manifest.csv",MANIFEST_SHA),
        "rededge_quality.csv":(root/V4_DIR/"cohort_706/rededge_quality.csv",QUALITY_SHA),
    }
    for name,(source,digest) in files.items():
        if file_sha256(source)!=digest:
            raise ValueError(f"Frozen v4 catalog source hash changed: {name}")
        destination=root/CATALOG/name
        if destination.exists() and file_sha256(destination)!=digest:
            raise ValueError(f"Frozen v4 catalog copy changed: {name}")
    for name,(source,digest) in files.items():
        destination=root/CATALOG/name
        if destination.exists():
            continue
        destination.parent.mkdir(parents=True,exist_ok=True)
        tmp=destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
        try:
            shutil.copyfile(source,tmp)
            if file_sha256(tmp)!=digest:
                raise ValueError(f"Frozen v4 catalog copy failed: {name}")
            tmp.replace(destination)
        finally:
            tmp.unlink(missing_ok=True)
    return root/CATALOG


def _load_catalog(root):
    directory=Path(root)/CATALOG
    manifest_path,quality_path=directory/"training_manifest.csv",directory/"rededge_quality.csv"
    if not manifest_path.exists() and not quality_path.exists():
        return None
    if not manifest_path.is_file() or not quality_path.is_file():
        raise ValueError("Frozen red-edge catalog incomplete")
    if file_sha256(manifest_path)!=MANIFEST_SHA or file_sha256(quality_path)!=QUALITY_SHA:
        raise ValueError("Frozen red-edge catalog hash mismatch")
    m=pd.read_csv(manifest_path,dtype={"sample_id":str})
    q=pd.read_csv(quality_path,dtype={"sample_id":str})
    if len(m)!=706 or len(q)!=706 or m.sample_id.duplicated().any() or not m.sample_id.equals(q.sample_id):
        raise ValueError("Frozen red-edge catalog row identity mismatch")
    if not m.sentinel_scene_id.equals(q.scene_id):
        raise ValueError("Frozen red-edge catalog scene mismatch")
    return m.merge(q,on="sample_id",validate="one_to_one")


def find_frozen_rededge(path, metadata, *, root=PROJECT_ROOT):
    """Prefer only a verified same-scene patch; missing catalog/match is not evidence."""
    catalog=_load_catalog(root)
    if catalog is None:
        return None
    digest=file_sha256(path)
    # The caller supplies verified sidecar coordinates. No untrusted sample-ID lookup.
    same_bytes=catalog.loc[catalog.original_patch_sha256==digest]
    if same_bytes.empty:
        return None
    candidates=same_bytes.loc[(same_bytes.scene_id==metadata["scene_id"])
                           &np.isclose(same_bytes.latitude,float(metadata["latitude"]),rtol=0,atol=1e-8)
                           &np.isclose(same_bytes.longitude,float(metadata["longitude"]),rtol=0,atol=1e-8)]
    if candidates.empty:
        raise ValueError("Known base patch does not match declared scene and coordinates")
    identity=["rededge_patch_sha256","scene_band_hash","rededge_joint_valid_fraction"]
    if len(candidates[identity].drop_duplicates())!=1:
        raise ValueError("Ambiguous frozen red-edge evidence for identical base patch")
    selected=candidates.sort_values("sample_id").iloc[0]
    sample=SimpleNamespace(sample_id=str(selected.sample_id),sentinel_scene_id=selected.scene_id,
                           latitude=float(selected.latitude),longitude=float(selected.longitude),
                           image_path=str(Path(path).resolve()))
    features,evidence=cached_rededge_sample(sample,output=Path(root)/V4_DIR/"patches",
                                            min_joint_valid=.05,allow_acquisition=False)
    for key in ("sample_id","scene_id","rededge_patch_sha256","original_patch_sha256","scene_band_hash"):
        if str(evidence[key])!=str(selected[key]):
            raise ValueError(f"Frozen red-edge evidence drift: {key}")
    if not np.isclose(evidence["rededge_joint_valid_fraction"],selected.rededge_joint_valid_fraction,rtol=0,atol=1e-12):
        raise ValueError("Frozen red-edge QA differs")
    return features,{**evidence,"rededge_cache_source":"verified_frozen_v4_706",
                     "frozen_manifest_sha256":MANIFEST_SHA,"frozen_quality_sha256":QUALITY_SHA}
