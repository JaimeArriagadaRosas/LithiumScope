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


from lithiumscope.model_2.data.expansion_vnext_build_manifest import ensure_expanded_m2_manifest

if __name__ == "__main__":
    main()
