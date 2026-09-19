from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path

import pandas as pd

from lithiumscope.core.config import load_config
from lithiumscope.core.logger import get_logger
from lithiumscope.core.paths import PROJECT_ROOT
from lithiumscope.model_2.data.sample_source import load_georeferenced_li_samples
from lithiumscope.model_2.data.sentinel2 import (
    Sentinel2Provider,
    SentinelAssetError,
    SentinelConfig,
    SentinelProviderError,
    SentinelSceneUnavailable,
    download_patch,
)
from lithiumscope.runtime.console_status import Spinner

logger = get_logger("model_2.dataset_builder")

_MANIFEST_COLUMNS = [
    "sample_id",
    "Li_icpms",
    "longitude",
    "latitude",
    "spatial_group",
    "image_path",
    "sentinel_scene_id",
    "sentinel_cloud_cover",
]
_FAILURE_COLUMNS = [
    "sample_id",
    "longitude",
    "latitude",
    "category",
    "detail",
]


@dataclass(frozen=True)
class Model2DatasetResult:
    manifest_path: Path
    total_candidates: int
    ready_samples: int
    failed_samples: int
    resumed_samples: int = 0


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
        request_timeout_seconds=float(
            imagery.get("request_timeout_seconds", 60)
        ),
        max_retries=int(imagery.get("max_retries", 3)),
        scene_cache_decimals=int(
            imagery.get("scene_cache_decimals", 4)
        ),
        patch_size_m=float(imagery["patch_size_m"]),
        patch_pixels=int(imagery["patch_pixels"]),
        bands=tuple(str(value) for value in imagery["bands"]),
    )


def _cache_name(
    sample_id: str,
    latitude: float,
    longitude: float,
) -> str:
    identity = (
        f"{sample_id}|{latitude:.6f}|{longitude:.6f}"
    ).encode("utf-8")
    suffix = sha1(identity, usedforsecurity=False).hexdigest()[:10]
    return f"{sample_id}_{suffix}.npy"


def _sample_key(
    sample_id: str,
    latitude: float,
    longitude: float,
) -> str:
    return f"{sample_id}|{latitude:.6f}|{longitude:.6f}"


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _load_valid_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=_MANIFEST_COLUMNS)
    try:
        frame = pd.read_csv(path)
    except Exception:
        logger.warning(
            "Could not read Model 2 checkpoint %s; ignoring it",
            path,
            exc_info=True,
        )
        return pd.DataFrame(columns=_MANIFEST_COLUMNS)

    if "image_path" not in frame.columns:
        return pd.DataFrame(columns=_MANIFEST_COLUMNS)

    valid = frame[
        frame["image_path"].map(
            lambda value: Path(str(value)).exists()
        )
    ].copy()
    for column in _MANIFEST_COLUMNS:
        if column not in valid.columns:
            valid[column] = pd.NA
    return valid[_MANIFEST_COLUMNS]


def _persist_progress(
    rows: dict[str, dict],
    failures: list[dict],
    partial_manifest_path: Path,
    failures_path: Path,
) -> None:
    manifest = pd.DataFrame(
        list(rows.values()),
        columns=_MANIFEST_COLUMNS,
    )
    failure_frame = pd.DataFrame(
        failures,
        columns=_FAILURE_COLUMNS,
    )
    _atomic_write_csv(manifest, partial_manifest_path)
    _atomic_write_csv(failure_frame, failures_path)


def _progress_message(
    done: int,
    total: int,
    ready: int,
    failed: int,
    resumed: int,
) -> str:
    return (
        f"Modelo 2 · Sentinel-2 {done}/{total} | "
        f"listos={ready} | omitidos={failed} | cache={resumed}"
    )


def ensure_model_2_dataset(
    model_1_dataset: Path,
    force: bool = False,
) -> Model2DatasetResult:
    config = load_config("model_2")
    training = config["training"]
    imagery = config["imagery"]

    manifest_path = _resolve_project_path(
        str(training["manifest_path"])
    )
    partial_manifest_path = _resolve_project_path(
        str(
            training.get(
                "partial_manifest_path",
                "data/processed/model_2/training_manifest.partial.csv",
            )
        )
    )
    failures_path = _resolve_project_path(
        str(
            training.get(
                "failures_path",
                "data/processed/model_2/training_failures.csv",
            )
        )
    )
    image_cache = _resolve_project_path(
        str(training["image_cache_dir"])
    )
    minimum_samples = int(training.get("minimum_samples", 50))
    checkpoint_every = max(
        1,
        int(imagery.get("checkpoint_every", 10)),
    )
    progress_every = max(
        1,
        int(imagery.get("progress_every", 25)),
    )
    max_consecutive_failures = max(
        1,
        int(imagery.get("max_consecutive_sample_failures", 10)),
    )

    if manifest_path.exists() and not force:
        manifest = _load_valid_manifest(manifest_path)
        if len(manifest) >= minimum_samples:
            logger.info(
                "Model 2 dataset already ready: %s (%d samples)",
                manifest_path,
                len(manifest),
            )
            return Model2DatasetResult(
                manifest_path,
                len(manifest),
                len(manifest),
                0,
                resumed_samples=len(manifest),
            )

    samples = load_georeferenced_li_samples(model_1_dataset)
    provider = Sentinel2Provider(_sentinel_config(config))

    checkpoint = pd.DataFrame(columns=_MANIFEST_COLUMNS)
    if not force:
        checkpoint = _load_valid_manifest(partial_manifest_path)

    rows: dict[str, dict] = {}
    for row in checkpoint.to_dict(orient="records"):
        key = _sample_key(
            str(row["sample_id"]),
            float(row["latitude"]),
            float(row["longitude"]),
        )
        rows[key] = row

    failures: list[dict] = []
    failed_samples = 0
    resumed_samples = len(rows)
    consecutive_failures = 0
    total = len(samples)

    spinner = Spinner(
        "Modelo 2 · validando catálogo Sentinel-2"
    ).start()

    try:
        provider.validate()
        spinner.update(
            _progress_message(
                0,
                total,
                len(rows),
                failed_samples,
                resumed_samples,
            )
        )

        for position, sample in samples.iterrows():
            sample_number = position + 1
            sample_id = str(sample["sample_id"])
            latitude = float(sample["latitude"])
            longitude = float(sample["longitude"])
            key = _sample_key(
                sample_id,
                latitude,
                longitude,
            )

            if key not in rows:
                destination = image_cache / _cache_name(
                    sample_id,
                    latitude,
                    longitude,
                )
                try:
                    patch_path, metadata = download_patch(
                        sample_id=sample_id,
                        latitude=latitude,
                        longitude=longitude,
                        destination=destination,
                        provider=provider,
                    )
                    rows[key] = {
                        "sample_id": sample_id,
                        "Li_icpms": float(sample["Li_icpms"]),
                        "longitude": longitude,
                        "latitude": latitude,
                        "spatial_group": str(sample["spatial_group"]),
                        "image_path": str(patch_path),
                        "sentinel_scene_id": metadata.get("scene_id"),
                        "sentinel_cloud_cover": metadata.get(
                            "cloud_cover"
                        ),
                    }
                    consecutive_failures = 0
                except SentinelProviderError:
                    logger.exception(
                        "Fatal Sentinel provider error; aborting acquisition"
                    )
                    raise
                except SentinelSceneUnavailable as exc:
                    failed_samples += 1
                    consecutive_failures = 0
                    failures.append(
                        {
                            "sample_id": sample_id,
                            "longitude": longitude,
                            "latitude": latitude,
                            "category": "no_scene",
                            "detail": str(exc),
                        }
                    )
                    logger.debug(
                        "No Sentinel scene sample=%s detail=%s",
                        sample_id,
                        exc,
                    )
                except SentinelAssetError as exc:
                    failed_samples += 1
                    consecutive_failures += 1
                    failures.append(
                        {
                            "sample_id": sample_id,
                            "longitude": longitude,
                            "latitude": latitude,
                            "category": "asset_error",
                            "detail": str(exc),
                        }
                    )
                    logger.debug(
                        "Sentinel asset failed sample=%s detail=%s",
                        sample_id,
                        exc,
                    )
                except Exception as exc:
                    failed_samples += 1
                    consecutive_failures += 1
                    failures.append(
                        {
                            "sample_id": sample_id,
                            "longitude": longitude,
                            "latitude": latitude,
                            "category": "unexpected_error",
                            "detail": str(exc),
                        }
                    )
                    logger.exception(
                        "Unexpected Sentinel acquisition failure sample=%s",
                        sample_id,
                    )

            spinner.update(
                _progress_message(
                    sample_number,
                    total,
                    len(rows),
                    failed_samples,
                    resumed_samples,
                )
            )

            if sample_number % checkpoint_every == 0:
                _persist_progress(
                    rows,
                    failures,
                    partial_manifest_path,
                    failures_path,
                )

            if (
                sample_number % progress_every == 0
                or sample_number == total
            ):
                logger.info(
                    "Sentinel progress=%d/%d ready=%d failed=%d resumed=%d",
                    sample_number,
                    total,
                    len(rows),
                    failed_samples,
                    resumed_samples,
                )

            if consecutive_failures >= max_consecutive_failures:
                raise RuntimeError(
                    "La adquisición Sentinel-2 acumuló "
                    f"{consecutive_failures} fallos de asset "
                    "consecutivos. Revise training_failures.csv."
                )

        manifest = pd.DataFrame(
            list(rows.values()),
            columns=_MANIFEST_COLUMNS,
        )
        if len(manifest) < minimum_samples:
            raise RuntimeError(
                "No fue posible construir un dataset espacial suficiente. "
                f"Disponibles={len(manifest)}, mínimo={minimum_samples}, "
                f"fallos={failed_samples}. Revise {failures_path}."
            )

        _atomic_write_csv(manifest, manifest_path)
    except (KeyboardInterrupt, SystemExit):
        spinner.stop("Modelo 2 · adquisición interrumpida")
        raise
    except Exception:
        spinner.fail(
            "Modelo 2 · preparación detenida; revise el diagnóstico"
        )
        raise
    finally:
        _persist_progress(
            rows,
            failures,
            partial_manifest_path,
            failures_path,
        )

    spinner.succeed(
        f"Modelo 2 · dataset listo: {len(rows)}/{total} muestras"
    )
    logger.info(
        "Model 2 manifest generated: %s ready=%d failed=%d resumed=%d",
        manifest_path,
        len(rows),
        failed_samples,
        resumed_samples,
    )
    return Model2DatasetResult(
        manifest_path,
        total,
        len(rows),
        failed_samples,
        resumed_samples=resumed_samples,
    )
