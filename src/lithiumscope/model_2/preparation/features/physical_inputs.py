"""Offline audited physical inputs for the experimental v5 representation."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from lithiumscope.core.hashing import file_sha256
from lithiumscope.datasets.management.sources.model_2_paths import resolve_image_path
from lithiumscope.model_2.preparation.features.physical_v5 import BANDS, physical_reflectance
from lithiumscope.model_2.preparation.features.spectral import DEFAULT_BAND_NAMES
from lithiumscope.model_2.preparation.features.rededge import NEW_BANDS
from lithiumscope.tools.acquisition.sentinel2.rededge import scene_item


def read_scene(path, scene_id):
    item=json.loads(Path(path).read_text(encoding="utf-8"))
    if item.get("id")!=scene_id or item.get("collection")!="sentinel-2-l2a":
        raise ValueError("STAC scene identity mismatch")
    metadata={}
    for band in BANDS:
        records=item.get("assets",{}).get(band,{}).get("raster:bands",[])
        if len(records)!=1 or not {"scale","offset","nodata"}<=set(records[0]):
            raise ValueError(f"Incomplete STAC radiometry: {band}")
        metadata[band]={k:records[0][k] for k in ("scale","offset","nodata")}
    return metadata


def load_physical_sample(row, *, base_sha, rededge_sha, scene_hash, scene_cache, rededge_cache):
    """Verify the exact frozen source bytes, identities and STAC radiometry."""
    scene_id=str(row.sentinel_scene_id)
    if not scene_id or "/" in scene_id or "\\" in scene_id:
        raise ValueError("Invalid STAC scene identity")
    base=resolve_image_path(row.image_path)
    if file_sha256(base)!=base_sha:
        raise ValueError("Base image changed")
    sidecar=base.with_suffix(".json")
    base_meta=json.loads(sidecar.read_text(encoding="utf-8"))
    coordinates=[float(row.latitude),float(row.longitude)]
    if (base_meta.get("scene_id")!=scene_id or base_meta.get("collection")!="sentinel-2-l2a"
            or tuple(base_meta.get("bands",[]))!=DEFAULT_BAND_NAMES
            or not np.allclose([float(base_meta["latitude"]),float(base_meta["longitude"])],
                               coordinates,rtol=0,atol=1e-8)):
        raise ValueError("Base sidecar does not match manifest")
    token=sha256(str(row.sample_id).encode()).hexdigest()[:22]
    rededge=Path(rededge_cache)/f"{token}.npy"
    red_meta_path=rededge.with_suffix(".json")
    red_meta=json.loads(red_meta_path.read_text(encoding="utf-8"))
    scene_path=Path(scene_cache)/f"{scene_id}.json"
    _, descriptor_hash=scene_item(scene_id,cache=scene_cache,allow_network=False)
    if descriptor_hash!=scene_hash or file_sha256(rededge)!=rededge_sha or red_meta.get("patch_sha256")!=rededge_sha:
        raise ValueError("Red-edge evidence changed")
    expected={"sample_id":str(row.sample_id),"source_image_sha256":base_sha,"scene_id":scene_id,
              "latitude":coordinates[0],"longitude":coordinates[1],"scene_band_hash":scene_hash,
              "band_order":list(NEW_BANDS),"patch_pixels":32,"patch_size_m":640}
    if red_meta.get("source")!=expected:
        raise ValueError("Red-edge provenance mismatch")
    original=np.load(base,allow_pickle=False)
    added=np.load(rededge,allow_pickle=False)
    if original.shape!=(6,32,32) or added.shape!=(4,32,32):
        raise ValueError("Unexpected base or red-edge geometry")
    raw=np.concatenate([original,added],axis=0)
    metadata=read_scene(scene_path,scene_id)
    physical=physical_reflectance(raw,metadata)
    return physical,{
        "base_image_sha256":base_sha,"rededge_image_sha256":rededge_sha,
        "base_sidecar_sha256":file_sha256(sidecar),"rededge_sidecar_sha256":file_sha256(red_meta_path),
        "scene_sha256":file_sha256(scene_path),"scene_band_hash":descriptor_hash,
        "declared_pixel_size_m":20,"geometry_certification":"historical_acquisition_contract_not_independent_georeference",
        "raw_joint_fraction":float(np.isfinite(raw).all(axis=0).mean()),
        "raw_finite_zero_pixels":int((np.isfinite(raw)&(raw==0)).sum()),
        "physical_joint_fraction":float(np.isfinite(physical).all(axis=0).mean()),
        "physical_negative_values":int((np.isfinite(physical)&(physical<0)).sum()),
        "physical_zero_values":int((np.isfinite(physical)&(physical==0)).sum()),
    }
