"""Isolated, resumable STAC red-edge acquisition for M2 v4.

Never rewrites a legacy cube or a frozen M2 feature cache. Every extra
band comes from the EXACT original scene ID, not a different/date scene.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from lithiumscope.core.paths import PROJECT_ROOT
from lithiumscope.model_2.data.sentinel2_patch_io import _read_patch
from lithiumscope.model_2.methodology.rededge_v4 import (
    NEW_BANDS,
    extract_rededge_v4,
    reflectance_from_dn,
)

STAC = "https://earth-search.aws.element84.com/v1/collections/sentinel-2-l2a/items"
SCENE_CACHE = (
    PROJECT_ROOT / "results/scientific_simulations/sentinel_methodology_20261112" / "inventory/stac_item_cache"
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scene_item(scene_id: str, *, cache: Path = SCENE_CACHE) -> tuple[dict, str]:
    if "/" in scene_id or "\\" in scene_id or not scene_id:
        raise ValueError("Invalid STAC scene identifier")
    path = cache / (scene_id + ".json")
    extra = PROJECT_ROOT / "results/scientific_simulations/m2_rededge_v4_20261113/stac_cache"
    target = extra / (scene_id + ".json")
    if path.is_file():
        item = json.loads(path.read_text(encoding="utf-8"))
    elif target.is_file():
        item = json.loads(target.read_text(encoding="utf-8"))
    else:
        import requests

        response = requests.get(f"{STAC}/{scene_id}", timeout=35)
        response.raise_for_status()
        item = response.json()
        # Store NEW scenes separately; never alter the 111-scene inventory.
        if item.get("id") != scene_id or item.get("collection") != "sentinel-2-l2a":
            raise ValueError("Unexpected STAC response")
        extra.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(item, indent=2), encoding="utf-8")
    if item.get("id") != scene_id or item.get("collection") != "sentinel-2-l2a":
        raise ValueError("STAC cache scene or collection mismatch")
    descriptor = {
        "scene_id": scene_id,
        "bands": {},
    }
    for name in NEW_BANDS:
        asset = item.get("assets", {}).get(name)
        if not isinstance(asset, dict):
            raise ValueError(f"Missing real Sentinel asset {name} in {scene_id}")
        raster = asset.get("raster:bands", [])
        if len(raster) != 1:
            raise ValueError("Missing per-band physical radiometric metadata")
        rs = raster[0]
        scale = float(rs["scale"])
        offset = float(rs["offset"])
        if not np.isfinite(scale) or scale <= 0 or not np.isfinite(offset):
            raise ValueError("Invalid STAC scale/offset")
        href = str(asset.get("href", ""))
        if not href.startswith("https://"):
            raise ValueError("Refusing non-HTTPS Sentinel asset")
        descriptor["bands"][name] = {
            "href": href,
            "scale": scale,
            "offset": offset,
            "nodata": rs.get("nodata"),
        }
    serial = json.dumps(descriptor, sort_keys=True, separators=(",", ":"))
    return descriptor, hashlib.sha256(serial.encode("utf-8")).hexdigest()


def cached_rededge_sample(
    sample,
    *,
    output: Path,
    min_joint_valid: float = 0.05,
) -> tuple[dict[str, float], dict]:
    """Load or acquire exact-scene four-band reflectance for one sample."""
    sample_id = str(sample.sample_id)
    scene_id = str(sample.sentinel_scene_id)
    original = Path(str(sample.image_path))
    if not original.is_file() or not original.stat().st_size:
        raise FileNotFoundError(f"Original six-band patch is missing: {sample_id}")
    if not (np.isfinite(float(sample.latitude)) and np.isfinite(float(sample.longitude))):
        raise ValueError("Invalid sample coordinates")
    descriptor, descriptor_hash = scene_item(scene_id)
    identity = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:22]
    output.mkdir(parents=True, exist_ok=True)
    patch_file = output / (identity + ".npy")
    meta_file = output / (identity + ".json")
    expected = {
        "sample_id": sample_id,
        "source_image_sha256": sha(original),
        "scene_id": scene_id,
        "latitude": float(sample.latitude),
        "longitude": float(sample.longitude),
        "scene_band_hash": descriptor_hash,
        "band_order": list(NEW_BANDS),
        "patch_pixels": 32,
        "patch_size_m": 640,
    }
    if patch_file.exists() or meta_file.exists():
        if not (patch_file.is_file() and meta_file.is_file()):
            raise ValueError("Incomplete red-edge cache; no silent refresh")
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        if meta.get("source") != expected or meta.get("patch_sha256") != sha(patch_file):
            raise ValueError(f"Red-edge cache drift detected: {sample_id}")
        dn = np.load(patch_file, allow_pickle=False)
    else:
        arrays = []
        for name in NEW_BANDS:
            arrays.append(
                _read_patch(
                    descriptor["bands"][name]["href"],
                    float(sample.latitude),
                    float(sample.longitude),
                    640,
                    32,
                )
            )
        dn = np.stack(arrays).astype(np.float32)
        if dn.shape != (4, 32, 32):
            raise ValueError("Unexpected four-band Sentinel patch dimension")
        temporary = patch_file.with_suffix(".tmp.npy")
        np.save(temporary, dn, allow_pickle=False)
        os.replace(temporary, patch_file)
        meta_file.write_text(
            json.dumps(
                {
                    "source": expected,
                    "patch_sha256": sha(patch_file),
                    "data_source": "Real Sentinel-2 L2A 20-m COG, exact scene",
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    scales = tuple(
        (
            descriptor["bands"][name]["scale"],
            descriptor["bands"][name]["offset"],
        )
        for name in NEW_BANDS
    )
    physical = reflectance_from_dn(dn, scales)
    features, coverage = extract_rededge_v4(
        physical,
        min_joint_valid=min_joint_valid,
    )
    return features, {
        "sample_id": sample_id,
        "scene_id": scene_id,
        "rededge_joint_valid_fraction": coverage,
        "rededge_patch_sha256": sha(patch_file),
        "original_patch_sha256": expected["source_image_sha256"],
        "scene_band_hash": descriptor_hash,
    }
