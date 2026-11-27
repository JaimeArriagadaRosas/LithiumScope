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
    # Frozen historical manifest: do not re-download, rebuild or overwrite it.
    if training.get("frozen_manifest_sha256"):
        from lithiumscope.datasets.management.sources.model_2 import require_frozen_manifest
        from lithiumscope.core.paths import PROJECT_ROOT
        if force:
            raise ValueError("Cannot rebuild a frozen M2 cohort")
        frozen = require_frozen_manifest(training, PROJECT_ROOT)
        count = len(pd.read_csv(frozen))
        return Model2DatasetResult(frozen, count, count, 0, resumed_samples=count)
    from lithiumscope.model_2.data.checkpoint_contract import (
        source_signature, acquisition_signature, reconcile_checkpoint, atomic_text, archive_failures,
    )
    import json
    from lithiumscope.tools.acquisition.sentinel2.registered_patch import validate_interval
    validate_interval(imagery["datetime"])
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
    current_source_sha256 = source_signature(file_sha256(model_1_dataset), imagery, training)
    samples = load_georeferenced_li_samples(model_1_dataset)
    atomic_text(manifest_path.with_suffix('.source-audit.json'), json.dumps(samples.attrs.get('source_reconciliation', []), ensure_ascii=False, indent=2))
    minimum_samples = int(training.get("minimum_samples", 50))
    checkpoint_every = max(1, int(imagery.get("checkpoint_every", 10)))
    progress_every = max(1, int(imagery.get("progress_every", 25)))
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
    source_matches = cached_source_sha256 == current_source_sha256
    if manifest_path.exists() and not force and source_matches:
        manifest = _load_valid_manifest(manifest_path)
        manifest, rejected = reconcile_checkpoint(manifest, samples, imagery)
        if len(manifest) >= minimum_samples and len(manifest) == len(samples) and not rejected:
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
        logger.info("Model 2 source or policy changed; reconcile cached imagery.")
    provider = Sentinel2Provider(_sentinel_config(config))
    checkpoint = pd.DataFrame(columns=_MANIFEST_COLUMNS)
    if not force:
        checkpoint = _load_valid_manifest(partial_manifest_path)
        if manifest_path.is_file():
            previous = _load_valid_manifest(manifest_path)
            if checkpoint.empty:
                checkpoint = previous
        checkpoint, rejected = reconcile_checkpoint(checkpoint, samples, imagery)
        atomic_text(manifest_path.with_suffix('.checkpoint-audit.json'), json.dumps(rejected, indent=2))
    archive_failures(failures_path)
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
                destination = image_cache / acquisition_signature(imagery) / _cache_name(
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
                        acquisition_signature=acquisition_signature(imagery),
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
        # Repeated/resumed acquisitions publish in the current source order.
        order = {sid: i for i, sid in enumerate(samples.sample_id)}
        manifest = manifest.sort_values("sample_id", key=lambda ids: ids.map(order)).reset_index(drop=True)
        _atomic_write_csv(manifest, manifest_path)
        atomic_text(source_fingerprint_path, current_source_sha256 + "\n")
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
    spinner.succeed(f"Modelo 2 · dataset listo: {len(rows)}/{total} muestras")
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
