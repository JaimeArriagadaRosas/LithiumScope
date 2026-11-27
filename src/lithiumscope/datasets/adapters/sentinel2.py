"""Source-level patch inspection without resampling, imputation or feature extraction."""
from pathlib import Path
import json
import numpy as np
from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.logging.session import get_logger

def inspect_sentinel_patch(path: Path, *, expected_bands, expected_pixels, expected_size_m):
    path = Path(path)
    if not expected_bands or len(set(expected_bands)) != len(expected_bands) or expected_pixels <= 0 or expected_size_m <= 0:
        raise ValueError("Invalid expected patch contract")
    sidecars = [p for p in (path.with_suffix(".json"), path.with_suffix(path.suffix + ".json")) if p.is_file()]
    if len(sidecars) != 1:
        raise ValueError("Exactly one patch metadata sidecar required")
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
            if ds.crs is None or not ds.crs.is_projected or not np.isclose(ds.crs.linear_units_factor[1], 1):
                raise ValueError("Projected metric CRS required")
            if tuple(ds.descriptions) != tuple(expected_bands):
                raise ValueError("Raster band descriptions disagree")
            size = expected_size_m / expected_pixels
            if not np.allclose([ds.transform.a, ds.transform.e, ds.transform.b, ds.transform.d],
                               [size, -size, 0, 0], rtol=0, atol=1e-5):
                raise ValueError("Raster geometry mismatch")
            data = ds.read(masked=True).astype(float).filled(np.nan)
        geometry = "metric_grid_verified_center_not_checked"
    else:
        raise ValueError("Unsupported patch format")
    if data.shape != (len(expected_bands), expected_pixels, expected_pixels) or not np.issubdtype(data.dtype, np.number) or np.iscomplexobj(data):
        raise ValueError("Invalid numeric patch shape")
    if file_sha256(path) != digest:
        raise ValueError("Patch changed during inspection")
    evidence = {"source_sha256": digest, "scene_id": meta["scene_id"], "geometry": geometry,
                "bands": list(expected_bands), "joint_valid_fraction": float(np.isfinite(data).all(axis=0).mean()),
                "band_valid_fractions": np.isfinite(data).mean(axis=(1,2)).tolist(),
                "training_label_certified": False}
    get_logger("datasets.adapters.sentinel2").info("Patch inspected scene=%s valid=%.4f", meta["scene_id"], evidence["joint_valid_fraction"])
    return data, evidence
