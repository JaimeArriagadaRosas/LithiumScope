"""Isolated responsibility with unchanged original API."""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lithiumscope.model_2.data.expansion_vnext import Path

from lithiumscope.core.paths import PROJECT_ROOT

def ensure_expanded_m2_manifest(
    root: Path = PROJECT_ROOT, *,
    acquire: bool = True, max_new_downloads: int | None = None,
    retry_failed: bool = False,
) -> Path | None:
    """Incrementally acquire requested samples and prepare isolated M2 manifest.

    acquire=False only selects/assesses; max_new_downloads bounds a prefetch.
    When called by the training menu it completes the requested target before
    permitting training, or fails clearly if too few acceptable samples remain.
    """
    from lithiumscope.model_2.data.expansion_vnext import PROJECT_ROOT, Path, Sentinel2Provider, SentinelAssetError, SentinelProviderError, SentinelSceneUnavailable, _cloud_mask, _sentinel_config, _signature, _validate_bands, atomic_csv, balanced_queue, candidates, download_patch, file_sha256, frozen_m2, json, load_config, np, pd, project_path, replace, source_audit
    config = load_config("model_2")
    settings = config["training"]["expansion"]
    if not settings.get("enabled", False):
        return project_path(root, config["training"]["manifest_path"])
    if max_new_downloads is not None and max_new_downloads < 0:
        raise ValueError("max_new_downloads cannot be negative")
    old_path = project_path(root, config["training"]["manifest_path"])
    old = frozen_m2(root, config)
    audit = source_audit(root, config)
    audit_path = project_path(root, settings["source_audit_path"])
    source_hash = _signature(old_path, audit_path, config)
    stage = project_path(root, settings["state_dir"])
    stage.mkdir(parents=True, exist_ok=True)
    selected_path = stage / "selected_candidate_queue.csv"
    selection_meta = stage / "selection_signature.json"
    full = candidates(
        audit, old, float(settings["max_li_ppm"]),
        float(config["training"]["spatial_group_degrees"]),
    )
    target = int(settings["target_new_samples"])
    minimum = int(settings["minimum_new_samples"])
    if len(full) < minimum:
        raise RuntimeError(
            f"Only {len(full)} provisional GEOROC candidates available; "
            f"minimum is {minimum}."
        )
    if (selected_path.is_file() and selection_meta.is_file()
            and json.loads(selection_meta.read_text(encoding="utf-8"))
                .get("signature") == source_hash):
        selected = pd.read_csv(selected_path)
    else:
        selected = balanced_queue(
            full, target, int(settings["max_per_publication"])
        )
        atomic_csv(selected, selected_path)
        selection_meta.write_text(
            json.dumps({"signature": source_hash, "eligible": len(full),
                        "queued": len(selected), "requested": target},
                       indent=2), encoding="utf-8"
        )
    print(
        f"M2 vNext: {len(full)} candidatos, {len(selected)} grupos "
        f"seleccionados, objetivo={target} etiquetas nuevas."
    )
    if selected.empty:
        raise RuntimeError("No diverse candidate groups found for M2")
    if not acquire:
        return None

    patch_dir = project_path(root, settings["patch_dir"])
    completed_path = stage / "valid_new_samples.csv"
    failures_path = stage / "download_failures.csv"
    columns = [
        "sample_id", "Li_icpms", "longitude", "latitude", "spatial_group",
        "image_path", "sentinel_scene_id", "sentinel_cloud_cover",
        "source_dataset", "source_citation", "coordinate_verification",
        "scl_cloud_fraction",
    ]
    completed = (
        pd.read_csv(completed_path)
        if completed_path.is_file()
        else pd.DataFrame(columns=columns)
    )
    completed = completed.loc[
        completed.image_path.map(lambda x: Path(str(x)).is_file())
    ].drop_duplicates("sample_id")
    # Prior failures are logged and retried on subsequent invocations.
    failures = (
        pd.read_csv(failures_path).to_dict("records")
        if failures_path.is_file()
        else []
    )
    seen = set(completed.sample_id.astype(str))
    # Network/cloud/asset failures are persisted. Do not re-query them on
    # every interactive training run unless the user explicitly retries.
    blocked = (
        set() if retry_failed
        else {str(item.get("sample_id")) for item in failures}
    )
    if not seen.issubset(set(selected.sample_id.astype(str))):
        raise RuntimeError("Existing sample cache does not match selected queue")
    ready = len(completed)
    done = 0
    provider = None
    consecutive_fails = 0
    for candidate in selected.itertuples(index=False):
        if ready >= target:
            break
        if candidate.sample_id in seen or candidate.sample_id in blocked:
            continue
        if max_new_downloads is not None and done >= max_new_downloads:
            break
        if provider is None:
            settings_cloud = float(settings["max_scene_cloud_percent"])
            sentinel_cfg = replace(
                _sentinel_config(config),
                cloud_cover_max=min(
                    float(config["imagery"]["cloud_cover_max"]),
                    settings_cloud,
                ),
            )
            provider = Sentinel2Provider(sentinel_cfg)
            provider.validate()
        done += 1
        path = patch_dir / f"{candidate.sample_id}.npy"
        try:
            scene = provider.search_best_scene(
                float(candidate.Latitude), float(candidate.Longitude)
            )
            qc = _cloud_mask(
                scene, float(candidate.Latitude), float(candidate.Longitude),
                provider, float(settings["max_patch_cloud_fraction"]),
            )
            patch, metadata = download_patch(
                sample_id=str(candidate.sample_id),
                latitude=float(candidate.Latitude),
                longitude=float(candidate.Longitude),
                destination=path,
                provider=provider,
            )
            _validate_bands(patch, config)
            record = {
                "sample_id": candidate.sample_id,
                "Li_icpms": float(candidate.Li_icpms),
                "longitude": float(candidate.Longitude),
                "latitude": float(candidate.Latitude),
                "spatial_group": str(candidate.spatial_group),
                "image_path": str(patch),
                "sentinel_scene_id": str(metadata.get("scene_id")),
                "sentinel_cloud_cover": float(metadata.get("cloud_cover")),
                "source_dataset": "GEOROC_B_SURFACE_PROVISIONAL",
                "source_citation": str(candidate.CITATIONS),
                "coordinate_verification": "PRE_SCREENED_NOT_FIELD_VERIFIED",
                "scl_cloud_fraction": qc["patch_cloud_fraction_scl"],
            }
            completed = pd.concat(
                [completed, pd.DataFrame([record])], ignore_index=True
            )
            seen.add(candidate.sample_id)
            ready += 1
            consecutive_fails = 0
            atomic_csv(completed[columns], completed_path)
            print(
                f"M2 DOWNLOAD {done}: usable={ready}/{target} "
                f"group={candidate.spatial_group} cloud="
                f"{qc['patch_cloud_fraction_scl']:.1%}",
                flush=True,
            )
        except SentinelProviderError:
            # Global provider failure: stop rather than silently training old.
            raise
        except (SentinelAssetError, SentinelSceneUnavailable,
                ValueError, OSError) as exc:
            consecutive_fails += 1
            failures.append({
                "sample_id": candidate.sample_id,
                "spatial_group": candidate.spatial_group,
                "problem": exc.__class__.__name__,
                "detail": str(exc)[:400],
            })
            atomic_csv(pd.DataFrame(failures), failures_path)
            print(f"M2 SKIPPED {candidate.sample_id}: {exc}", flush=True)
            if path.is_file():
                path.unlink()
                sidecar = path.with_suffix(".json")
                if sidecar.exists():
                    sidecar.unlink()
            if consecutive_fails >= int(settings["max_consecutive_failures"]):
                raise RuntimeError("Too many consecutive Sentinel download failures")
    # Bounded prefetch may intentionally stop before meeting the threshold.
    if max_new_downloads is not None and ready < minimum:
        print(
            f"M2 PREFETCH: {ready} new patches cached, at least {minimum} "
            "are required before producing a training manifest.",
            flush=True,
        )
        return None
    if ready < minimum:
        raise RuntimeError(
            f"M2 vNext has only {ready} quality-screened patches; "
            f"at least {minimum} required. No fallback to old M2."
        )
    extended = completed.head(target).copy()
    if extended.spatial_group.duplicated().any():
        raise RuntimeError("Expanded samples repeat their own spatial groups")
    if set(extended.spatial_group).intersection(set(old.spatial_group)):
        raise RuntimeError("New and historical M2 spatial groups overlap")
    old_copy = old.copy()
    old_copy["source_dataset"] = "MAMANI09_HISTORICAL"
    old_copy["source_citation"] = ""
    old_copy["coordinate_verification"] = "LEGACY_HISTORICAL"
    old_copy["scl_cloud_fraction"] = np.nan
    combined = pd.concat(
        [old_copy, extended], ignore_index=True, sort=False
    )
    if combined.sample_id.duplicated().any():
        raise RuntimeError("Duplicate sample IDs in M2 expansion")
    from lithiumscope.model_2.training.scientific_validation import (
        make_scientific_folds,
    )
    splits = make_scientific_folds(
        combined.Li_icpms, combined.spatial_group,
        n_splits=int(config["validation"]["folds"]),
        quantile=float(config["model"]["high_lithium_quantile"]),
    )
    destination = project_path(root, settings["output_manifest_path"])
    atomic_csv(combined, destination)
    summary = {
        "manifest": str(destination),
        "original_legacy_count": len(old),
        "new_georoc_count": len(extended),
        "total": len(combined),
        "spatial_groups_old": int(old.spatial_group.nunique()),
        "spatial_groups_new": int(extended.spatial_group.nunique()),
        "source_audit_sha256": file_sha256(audit_path),
        "legacy_sha256": file_sha256(old_path),
        "new_manifest_sha256": file_sha256(destination),
        "georoc_label_caution": "PRE_SCREENED_NOT_FIELD_VERIFIED",
        "folds": len(splits),
        "trained": False,
    }
    (stage / "expansion_summary.json").write_text(
        json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8"
    )
    print("M2 EXPANSION READY",json.dumps(summary,ensure_ascii=False),flush=True)
    return destination
