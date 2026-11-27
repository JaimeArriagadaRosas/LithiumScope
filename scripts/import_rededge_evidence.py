"""Copy only the frozen cohort's red-edge cubes and exact STAC descriptors."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from uuid import uuid4

import pandas as pd

from lithiumscope.core.hashing import file_sha256
from lithiumscope.datasets.management.sources.model_2_paths import resolve_image_path

ROOT = Path(__file__).resolve().parents[1]
V4 = Path("results/scientific_simulations/m2_rededge_v4_20261113")
STAC = Path("results/scientific_simulations/sentinel_methodology_20261112/inventory/stac_item_cache")


def import_evidence(source: Path, root: Path) -> dict:
    source, root = source.resolve(), root.resolve()
    if source == root or source.is_relative_to(root) or root.is_relative_to(source):
        raise ValueError("Independent historical and destination roots required")
    manifest = pd.read_csv(root/"data/processed/model_2/training_manifest.csv")
    quality = pd.read_csv(root/V4/"cohort_706/rededge_quality.csv")
    if len(manifest) != 706 or len(quality) != 706 or manifest.sample_id.duplicated().any():
        raise ValueError("Expected unique frozen cohort of 706")
    if manifest.sample_id.astype(str).tolist() != quality.sample_id.astype(str).tolist():
        raise ValueError("Manifest/QA order mismatch")
    artifacts = {}
    def register(relative: Path, digest: str | None = None):
        original = (source/relative).resolve()
        if not original.is_relative_to(source) or not original.is_file():
            raise FileNotFoundError(original)
        actual = file_sha256(original)
        if digest and actual != digest:
            raise ValueError(f"Historical hash mismatch: {relative}")
        artifacts[relative.as_posix()] = actual

    for row, qa in zip(manifest.itertuples(index=False), quality.itertuples(index=False), strict=True):
        token = hashlib.sha256(str(row.sample_id).encode()).hexdigest()[:22]
        patch, sidecar = V4/"patches"/f"{token}.npy", V4/"patches"/f"{token}.json"
        register(patch, qa.rededge_patch_sha256)
        register(sidecar)
        meta = json.loads((source/sidecar).read_text(encoding="utf-8"))
        identity = meta["source"]
        expected = {"sample_id": str(row.sample_id), "scene_id": row.sentinel_scene_id,
                    "source_image_sha256": qa.original_patch_sha256, "scene_band_hash": qa.scene_band_hash,
                    "latitude": float(row.latitude), "longitude": float(row.longitude),
                    "band_order": ["rededge1","rededge2","rededge3","nir08"],
                    "patch_pixels": 32, "patch_size_m": 640}
        if identity != expected or meta["patch_sha256"] != qa.rededge_patch_sha256:
            raise ValueError(f"Red-edge sidecar identity mismatch: {row.sample_id}")
        if file_sha256(resolve_image_path(row.image_path, root=root)) != qa.original_patch_sha256:
            raise ValueError(f"Local base patch changed: {row.sample_id}")
        scene = STAC/f"{row.sentinel_scene_id}.json"
        if not (source/scene).is_file():
            scene = V4/"stac_cache"/f"{row.sentinel_scene_id}.json"
        if scene.as_posix() not in artifacts:
            register(scene)

    for relative, digest in artifacts.items():
        target = (root/relative).resolve()
        if not target.is_relative_to(root):
            raise ValueError("Target escapes workspace")
        if target.exists() and file_sha256(target) != digest:
            raise FileExistsError(f"Local artifact differs: {relative}")
    copied = 0
    for relative, digest in artifacts.items():
        target = root/relative
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
        try:
            shutil.copyfile(source/relative,temp)
            if file_sha256(temp) != digest:
                raise ValueError("Copy hash mismatch")
            temp.replace(target)
            copied += 1
        finally:
            temp.unlink(missing_ok=True)
    from lithiumscope.model_2.prediction.frozen_rededge import register_frozen_catalog
    catalog=register_frozen_catalog(root)
    report = {"frozen_catalog":str(catalog.relative_to(root)),"cohort": 706, "verified_files": len(artifacts), "copied": copied,
              "source_root": str(source), "files": artifacts}
    path = root/"results/diagnostics/rededge_import.json"
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2),encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source",type=Path)
    args = parser.parse_args()
    result = import_evidence(args.source,ROOT)
    print({k:v for k,v in result.items() if k!="files"})
