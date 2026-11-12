"""Opt-in, resumable GEOROC-to-Sentinel-2 extension, isolated to Model 2.

GEOROC Li supplies point labels, never spectral predictors. Source quality
screening is provisional: it cannot establish GPS accuracy from publications.
Neither legacy M2 nor the M1 release candidate is modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from lithiumscope.core.config import load_config
from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.paths import PROJECT_ROOT
from lithiumscope.model_2.data.builder import _sentinel_config
from lithiumscope.model_2.data.sample_source import _spatial_group
from lithiumscope.model_2.data.sentinel2 import (
    Sentinel2Provider, SentinelAssetError, SentinelProviderError,
    SentinelSceneUnavailable, _asset_href, _read_patch, download_patch,
)
from lithiumscope.model_2.steps.step_01_load_imagery import load_imagery
from lithiumscope.model_2.steps.step_04_image_preprocessing import preprocess_image
from lithiumscope.model_2.steps.step_05_spectral_features import extract_spectral_features


def project_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def atomic_csv(frame: pd.DataFrame, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(target)


def frozen_m2(root: Path, config: dict) -> pd.DataFrame:
    path = project_path(root, config["training"]["manifest_path"])
    training = config["training"]
    if not path.is_file() or file_sha256(path) != training["frozen_manifest_sha256"]:
        raise RuntimeError("The frozen 706-row M2 baseline is missing or changed")
    frame = pd.read_csv(path)
    if len(frame) != int(training["frozen_expected_rows"]):
        raise ValueError("Unexpected frozen M2 sample count")
    required = {"sample_id", "image_path", "Li_icpms", "latitude",
                "longitude", "spatial_group"}
    if not required.issubset(frame.columns):
        raise ValueError("Frozen M2 manifest lacks required fields")
    if not frame.image_path.map(lambda x: Path(str(x)).is_file()).all():
        raise FileNotFoundError("Existing M2 images are missing")
    return frame


def source_audit(root: Path, config: dict) -> pd.DataFrame:
    """Use the existing GEOROC screen or recreate it in M2-only staging."""
    expanded = config["training"]["expansion"]
    audit = project_path(root, expanded["source_audit_path"])
    if audit.is_file():
        return pd.read_csv(audit)
    from lithiumscope.datasets.georoc_filtered_acquisition import (
        acquire_filtered_georoc, GEOROC_FILTERED_NAME,
    )
    from lithiumscope.datasets.georoc_ingestion import ingest_georoc_files
    from lithiumscope.experiments.analysis.audit_georoc_source_references import (
        classify_record, decimal_places, parse_bibliography,
    )
    from lithiumscope.experiments.analysis.georoc_qc_protocols import protocol_masks

    staged = project_path(root, expanded["state_dir"])
    staged.mkdir(parents=True, exist_ok=True)
    raw = root / "data/raw/model_1/georoc" / GEOROC_FILTERED_NAME
    if not raw.is_file():
        raw = staged / GEOROC_FILTERED_NAME
        acquire_filtered_georoc(destination=raw)
    harmonized = staged / "georoc_feot_B_harmonized.csv"
    ingest_georoc_files([raw], harmonized, iron_total_mode="feot_equivalent")
    data = pd.read_csv(harmonized, low_memory=False)
    mask = protocol_masks(data)["B_reconstructed"]
    data = data.loc[mask & data.source_dataset.eq("GEOROC")].copy()
    original = pd.read_csv(raw, dtype=str, low_memory=False)
    original = original.rename(columns={
        "SAMPLE NAME": "source_sample", "LATITUDE MIN": "orig_lat",
        "LONGITUDE MIN": "orig_lon", "LOCATION COMMENT": "orig_comment",
    })
    keys = ["CITATIONS", "source_sample"]
    unique = original[keys + ["orig_lat", "orig_lon", "orig_comment"]]
    counts = original.groupby(keys).size().rename("raw_matches").reset_index()
    data = data.merge(unique.drop_duplicates(keys), on=keys,
                      how="left", validate="many_to_one")
    data = data.merge(counts, on=keys, how="left", validate="many_to_one")
    refs_path = staged / "georoc_reference_list.csv"
    if not refs_path.is_file():
        import requests
        reply = requests.get("https://georoc.eu/georoc/reference_list.csv",
                             timeout=40)
        reply.raise_for_status()
        if len(reply.content) < 100000:
            raise RuntimeError("Invalid GEOROC bibliography download")
        refs_path.write_bytes(reply.content)
    refs = parse_bibliography(refs_path.read_bytes().decode(
        "cp1252", errors="replace"
    ))
    sites = data.Latitude.round(5).astype(str) + "|" + data.Longitude.round(5).astype(str)
    data["site_count"] = sites.map(sites.value_counts())
    data["decimal_lat"] = data.orig_lat.map(decimal_places)
    data["decimal_lon"] = data.orig_lon.map(decimal_places)
    data["references_complete"] = data.CITATIONS.map(
        lambda x: bool(re.findall(r"\[(\d+)\]", str(x))) and all(
            k in refs for k in re.findall(r"\[(\d+)\]", str(x))
        )
    )
    data["screening_tier"] = [
        classify_record(bool(ref), int(lat), int(lon), int(site), str(comment),
                        bool(ambiguous))
        for ref, lat, lon, site, comment, ambiguous in zip(
            data.references_complete, data.decimal_lat, data.decimal_lon,
            data.site_count, data.orig_comment.fillna(""), data.raw_matches.gt(1),
            strict=True,
        )
    ]
    data["sample_id"] = [
        stable_sample_id(ref, name, lat, lon)
        for ref, name, lat, lon in zip(
            data.CITATIONS, data.source_sample, data.Latitude, data.Longitude,
            strict=True,
        )
    ]
    keep = ["sample_id", "CITATIONS", "source_sample", "Latitude",
            "Longitude", "Li_icpms", "LOCATION", "screening_tier",
            "references_complete", "site_count", "decimal_lat", "decimal_lon"]
    result = data[keep].drop_duplicates("sample_id")
    atomic_csv(result, audit)
    return result


def stable_sample_id(ref, name, lat, lon) -> str:
    key = f"GEOROC|{ref}|{name}|{float(lat):.6f}|{float(lon):.6f}"
    return "GEOROC_" + hashlib.sha256(key.encode()).hexdigest()[:16]


def candidates(audit: pd.DataFrame, old: pd.DataFrame,
               max_li_ppm: float, degrees: float) -> pd.DataFrame:
    required = {"sample_id", "CITATIONS", "Latitude", "Longitude",
                "Li_icpms", "LOCATION", "screening_tier"}
    if not required.issubset(audit.columns):
        raise ValueError(f"GEOROC screen missing {required - set(audit)}")
    frame = audit.loc[
        audit.screening_tier.eq("PROVISIONAL_SITE_REVIEW")
    ].copy()
    for column in ("Li_icpms", "Latitude", "Longitude"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.loc[
        frame.Li_icpms.between(0, max_li_ppm)
        & frame.Latitude.between(-90, 90)
        & frame.Longitude.between(-180, 180)
    ].copy()
    frame["spatial_group"] = [
        _spatial_group(lat, lon, degrees)
        for lat, lon in zip(frame.Latitude, frame.Longitude, strict=True)
    ]
    frame = frame.loc[
        ~frame.spatial_group.isin(set(old.spatial_group.astype(str)))
    ].copy()
    frame["country"] = frame.LOCATION.fillna("").astype(str).str.split(
        "/"
    ).str[2].str.strip().str.upper()
    frame["country"] = frame.country.replace("", "UNKNOWN")
    frame["high_lithium"] = frame.Li_icpms.ge(
        float(old.Li_icpms.quantile(.75))
    )
    frame["priority"] = frame.sample_id.astype(str).map(
        lambda s: hashlib.sha256(s.encode()).hexdigest()
    )
    return frame.drop_duplicates("sample_id").sort_values("priority")


def balanced_queue(frame: pd.DataFrame, target: int,
                   max_per_paper: int = 4) -> pd.DataFrame:
    """Rotate across countries and lower/higher lithium with unique locations."""
    if target <= 0 or max_per_paper <= 0:
        raise ValueError("Invalid selection bounds")
    chosen, groups, papers = [], set(), Counter()
    used = set()
    countries = sorted(frame.country.unique())
    while len(chosen) < min(3 * target, len(frame)):
        changed = False
        for country in countries:
            for high in (False, False, True):
                subset = frame.loc[
                    frame.country.eq(country) & frame.high_lithium.eq(high)
                ]
                for index, row in subset.iterrows():
                    if (index in used or row.spatial_group in groups
                            or papers[str(row.CITATIONS)] >= max_per_paper):
                        continue
                    used.add(index)
                    groups.add(row.spatial_group)
                    papers[str(row.CITATIONS)] += 1
                    chosen.append(index)
                    changed = True
                    break
                if len(chosen) >= 3 * target:
                    break
            if len(chosen) >= 3 * target:
                break
        if not changed:
            break
    selected = frame.loc[chosen].reset_index(drop=True)
    if selected.spatial_group.duplicated().any():
        raise RuntimeError("Repeated spatial groups")
    return selected



def _cloud_mask(scene: dict, latitude: float, longitude: float,
                provider: Sentinel2Provider, max_cloud: float) -> dict:
    """Check cloud/shadow/no-data inside the actual patch, not only scene EO%."""
    cfg = provider.config
    if "scl" not in scene.get("assets", {}):
        raise SentinelAssetError("Scene lacks SCL local cloud classification")
    scl = _read_patch(
        _asset_href(scene, "scl"), latitude, longitude,
        cfg.patch_size_m, cfg.patch_pixels, categorical=True,
    )
    finite = np.isfinite(scl)
    if finite.mean() < .95:
        raise SentinelAssetError("SCL patch contains more than 5% no-data")
    scl = np.rint(scl[finite]).astype(int)
    if (scl <= 1).mean() > .05:
        raise SentinelAssetError("SCL patch outside scene or unusable data")
    fraction = float(np.isin(scl, [3, 8, 9, 10, 11]).mean())
    if fraction > max_cloud:
        raise SentinelAssetError(
            f"Cloud/shadow/snow covers {fraction:.1%} of target patch "
            f"(maximum {max_cloud:.1%})"
        )
    return {"patch_cloud_fraction_scl": fraction,
            "patch_finite_scl_fraction": float(finite.mean())}


def _validate_bands(path: Path, config: dict) -> int:
    image = load_imagery(path)
    if image.shape != (6, 32, 32):
        raise ValueError(f"Unexpected Sentinel spectral shape: {image.shape}")
    if not np.all(np.isfinite(image).mean(axis=(1, 2)) >= .95):
        raise ValueError("More than 5% invalid pixels in a band")
    vec = extract_spectral_features(
        preprocess_image(image, normalize_per_band=False),
        band_names=tuple(config["imagery"]["bands"]),
    )
    if len(vec) != 126 or not np.isfinite(list(vec.values())).all():
        raise ValueError("Spectral feature extractor mismatch")
    return len(vec)


def _signature(baseline_path: Path, audit_path: Path, config: dict) -> str:
    expansion = config["training"]["expansion"]
    payload = {
        "baseline_sha": file_sha256(baseline_path),
        "audit_sha": file_sha256(audit_path),
        "target": int(expansion["target_new_samples"]),
        "max_li_ppm": float(expansion["max_li_ppm"]),
        "max_per_paper": int(expansion["max_per_publication"]),
        "spatial_degrees": float(config["training"]["spatial_group_degrees"]),
        "bands": list(config["imagery"]["bands"]),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--acquire", action="store_true")
    parser.add_argument("--max-new-downloads", type=int, default=None)
    parser.add_argument(
        "--retry-failed", action="store_true",
        help="Explicitly retry previously cloud-rejected/unavailable scenes",
    )
    args = parser.parse_args()
    ensure_expanded_m2_manifest(
        acquire=args.acquire, max_new_downloads=args.max_new_downloads,
        retry_failed=args.retry_failed,
    )


if __name__ == "__main__":
    main()
