from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path

import pandas as pd
import requests

from lithiumscope.core.config import load_config
from lithiumscope.core.logger import get_logger
from lithiumscope.core.paths import PROJECT_ROOT
from lithiumscope.model_2.data.sample_source import load_georeferenced_li_samples
from lithiumscope.model_2.data.sentinel2 import SentinelConfig, download_patch

logger = get_logger("model_2.dataset_builder")


@dataclass(frozen=True)
class Model2DatasetResult:
    manifest_path: Path
    total_candidates: int
    ready_samples: int
    failed_samples: int


def _resolve_project_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _sentinel_config(config: dict) -> SentinelConfig:
    imagery = config["imagery"]
    return SentinelConfig(
        stac_url=str(imagery["stac_url"]),
        collection=str(imagery["collection"]),
        datetime=str(imagery["datetime"]),
        cloud_cover_max=float(imagery["cloud_cover_max"]),
        search_limit=int(imagery["search_limit"]),
        patch_size_m=float(imagery["patch_size_m"]),
        patch_pixels=int(imagery["patch_pixels"]),
        bands=tuple(str(value) for value in imagery["bands"]),
    )


def _cache_name(sample_id: str, latitude: float, longitude: float) -> str:
    identity = f"{sample_id}|{latitude:.6f}|{longitude:.6f}".encode("utf-8")
    suffix = sha1(identity, usedforsecurity=False).hexdigest()[:10]
    return f"{sample_id}_{suffix}.npy"


def ensure_model_2_dataset(model_1_dataset: Path, force: bool = False) -> Model2DatasetResult:
    config = load_config("model_2")
    training = config["training"]
    manifest_path = _resolve_project_path(str(training["manifest_path"]))
    image_cache = _resolve_project_path(str(training["image_cache_dir"]))
    minimum_samples = int(training.get("minimum_samples", 50))

    if manifest_path.exists() and not force:
        manifest = pd.read_csv(manifest_path)
        if "image_path" in manifest.columns:
            valid = manifest[manifest["image_path"].map(lambda value: Path(value).exists())]
            if len(valid) >= minimum_samples:
                logger.info(
                    "Model 2 dataset already ready: %s (%d samples)",
                    manifest_path,
                    len(valid),
                )
                return Model2DatasetResult(
                    manifest_path,
                    len(manifest),
                    len(valid),
                    len(manifest) - len(valid),
                )

    samples = load_georeferenced_li_samples(model_1_dataset)
    sentinel = _sentinel_config(config)
    session = requests.Session()
    rows: list[dict] = []
    failures = 0

    print(f"  Preparando automáticamente Modelo 2: {len(samples)} muestras georreferenciadas")
    print("  Fuente de imágenes: Sentinel-2 L2A / Earth Search")

    for position, sample in samples.iterrows():
        sample_number = position + 1
        sample_id = str(sample["sample_id"])
        latitude = float(sample["latitude"])
        longitude = float(sample["longitude"])
        destination = image_cache / _cache_name(sample_id, latitude, longitude)
        try:
            patch_path, metadata = download_patch(
                sample_id=sample_id,
                latitude=latitude,
                longitude=longitude,
                destination=destination,
                config=sentinel,
                session=session,
            )
            rows.append(
                {
                    "sample_id": sample_id,
                    "Li_icpms": float(sample["Li_icpms"]),
                    "longitude": longitude,
                    "latitude": latitude,
                    "spatial_group": str(sample["spatial_group"]),
                    "image_path": str(patch_path),
                    "sentinel_scene_id": metadata.get("scene_id"),
                    "sentinel_cloud_cover": metadata.get("cloud_cover"),
                }
            )
            print(f"    [{sample_number}/{len(samples)}] {sample_id}: OK")
        except Exception as exc:
            failures += 1
            logger.warning("Sentinel acquisition failed sample=%s: %s", sample_id, exc)
            print(f"    [{sample_number}/{len(samples)}] {sample_id}: omitida ({exc})")

    manifest = pd.DataFrame(rows)
    if len(manifest) < minimum_samples:
        raise RuntimeError(
            "No fue posible construir un dataset espacial suficiente para Modelo 2. "
            f"Disponibles={len(manifest)}, mínimo={minimum_samples}, fallos={failures}. "
            "Revise conexión a Internet y logs de preboot."
        )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)
    logger.info(
        "Model 2 manifest generated automatically: %s ready=%d failed=%d",
        manifest_path,
        len(manifest),
        failures,
    )
    return Model2DatasetResult(manifest_path, len(samples), len(manifest), failures)
