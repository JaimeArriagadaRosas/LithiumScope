"""Source-level patch inspection without resampling, imputation or feature extraction."""
from pathlib import Path
import json
import numpy as np
from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.logging.session import get_logger

def validate_metric_raster(dataset):
    crs=dataset.crs
    if crs is None or not crs.is_projected:
        raise ValueError("Raster must use a projected metric CRS")
    _, factor=crs.linear_units_factor
    if not np.isclose(factor,1.,rtol=0,atol=1e-9):
        raise ValueError("Raster linear units must be metres")
    affine=dataset.transform
    if not np.isclose(affine.b,0,atol=1e-9) or not np.isclose(affine.d,0,atol=1e-9) or affine.a<=0 or affine.e>=0:
        raise ValueError("Raster must be north-up, with nonrotated positive pixel width")
    return affine


def inspect_sentinel_patch(path: Path, *, expected_bands, expected_pixels, expected_size_m):
    path = Path(path)
    if not expected_bands or len(set(expected_bands)) != len(expected_bands) or expected_pixels <= 0 or expected_size_m <= 0:
        raise ValueError("Invalid expected patch contract")
    sidecars = [p for p in (path.with_suffix(".json"), path.with_suffix(path.suffix + ".json")) if p.is_file()]
    if len(sidecars) != 1:
        raise ValueError("Exactly one patch metadata sidecar required")
    sidecar_digest = file_sha256(sidecars[0])
    meta = json.loads(sidecars[0].read_text(encoding="utf-8"))
    digest = file_sha256(path)
    if meta.get("image_sha256") != digest or meta.get("collection") != "sentinel-2-l2a":
        raise ValueError("Patch content hash or collection mismatch")
    if tuple(meta.get("bands", ())) != tuple(expected_bands) or meta.get("patch_pixels") != expected_pixels or meta.get("patch_size_m") != expected_size_m:
        raise ValueError("Patch metadata contract mismatch")
    if not isinstance(meta.get("scene_id"), str) or not meta["scene_id"].strip():
        raise ValueError("Missing scene identity")
    if path.suffix.lower() == ".npy":
        data = np.load(path, allow_pickle=False)
        geometry = "declared_only"
    elif path.suffix.lower() in {".tif", ".tiff"}:
        import rasterio
        with rasterio.open(path) as ds:
            affine = validate_metric_raster(ds)
            if tuple(ds.descriptions) != tuple(expected_bands):
                raise ValueError("Raster band descriptions disagree")
            size = expected_size_m / expected_pixels
            if not np.allclose([affine.a, affine.e, affine.b, affine.d],
                               [size, -size, 0, 0], rtol=0, atol=1e-5):
                raise ValueError("Raster geometry mismatch")
            data = ds.read(masked=True).astype(float).filled(np.nan)
        geometry = "metric_grid_verified_center_not_checked"
    else:
        raise ValueError("Unsupported patch format")
    if data.shape != (len(expected_bands), expected_pixels, expected_pixels) or not np.issubdtype(data.dtype, np.number) or np.iscomplexobj(data):
        raise ValueError("Invalid numeric patch shape")
    mask_evidence = {"state": "not_declared"}
    if meta.get("scl_sha256") is not None:
        mask_path = path.with_suffix(".scl.npy")
        mask_digest = file_sha256(mask_path)
        if mask_digest != meta["scl_sha256"]:
            raise ValueError("SCL mask hash mismatch")
        mask = np.load(mask_path, allow_pickle=False)
        if mask.shape != (expected_pixels, expected_pixels) or not np.issubdtype(mask.dtype, np.number) or np.iscomplexobj(mask):
            raise ValueError("Invalid SCL mask shape or dtype")
        valid = np.isfinite(mask) & (mask == np.rint(mask)) & (mask >= 0) & (mask <= 11)
        if file_sha256(mask_path) != mask_digest:
            raise ValueError("SCL mask changed during inspection")
        mask_evidence = {"state": "bytes_verified", "sha256": mask_digest,
                         "classified_fraction": float(valid.mean()),
                         "scientific_cloud_policy_certified": False}
    if file_sha256(path) != digest or file_sha256(sidecars[0]) != sidecar_digest:
        raise ValueError("Patch changed during inspection")
    evidence = {"source_sha256": digest, "sidecar_sha256": sidecar_digest, "scene_id": meta["scene_id"], "geometry": geometry,
                "bands": list(expected_bands), "joint_valid_fraction": float(np.isfinite(data).all(axis=0).mean()),
                "band_valid_fractions": np.isfinite(data).mean(axis=(1,2)).tolist(),
                "scl": mask_evidence, "training_label_certified": False}
    get_logger("datasets.adapters.sentinel2").info("Patch inspected scene=%s valid=%.4f", meta["scene_id"], evidence["joint_valid_fraction"])
    return data, evidence
