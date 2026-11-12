"""Isolated responsibility with unchanged original API."""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lithiumscope.model_2.data.builder import Model2DatasetResult, Path


def ensure_model_2_dataset(
    model_1_dataset: Path,
    force: bool = False,
) -> Model2DatasetResult:
    from lithiumscope.model_2.data.builder import Model2DatasetResult, Path, Sentinel2Provider, SentinelAssetError, SentinelProviderError, SentinelSceneUnavailable, Spinner, _MANIFEST_COLUMNS, _atomic_write_csv, _cache_name, _load_valid_manifest, _persist_progress, _progress_message, _resolve_project_path, _sample_key, _sentinel_config, download_patch, file_sha256, load_config, load_georeferenced_li_samples, logger, pd
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
    source_fingerprint_path = manifest_path.with_suffix(
        manifest_path.suffix + ".source.sha256"
    )
    current_source_sha256 = file_sha256(model_1_dataset)
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
    cached_source_sha256 = (
        source_fingerprint_path.read_text(
            encoding="utf-8"
        ).strip()
        if source_fingerprint_path.is_file()
        else ""
    )
    source_matches = (
        cached_source_sha256 == current_source_sha256
    )
    if manifest_path.exists() and not force and source_matches:
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
    if manifest_path.exists() and not source_matches:
        logger.info(
            "Model 2 source dataset changed; rebuilding manifest "
            "while reusing cached imagery when possible."
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
        source_fingerprint_path.write_text(
            current_source_sha256 + "\n",
            encoding="utf-8",
        )
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
