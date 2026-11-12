"""Bounded, opt-in GEOROC -> Sentinel-2 labelled patch technical pilot.

This module is strictly experimental. Neither the production Model 2 sample
loader, the training menu, canonical datasets, nor installed models are changed.
GEOROC supplies Li reference labels ONLY, never spectral predictors.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lithiumscope.core.config import load_config
from lithiumscope.core.paths import PROJECT_ROOT, RESULTS_DIR
from lithiumscope.experiments.analysis.georoc_qc_protocols import protocol_masks
from lithiumscope.model_2.data.builder import _sentinel_config
from lithiumscope.model_2.data.sample_source import _spatial_group
from lithiumscope.model_2.data.sentinel2 import (
    Sentinel2Provider, SentinelAssetError, SentinelSceneUnavailable,
    SentinelProviderError, download_patch,
)
from lithiumscope.model_2.steps.step_01_load_imagery import load_imagery
from lithiumscope.model_2.steps.step_04_image_preprocessing import preprocess_image
from lithiumscope.model_2.steps.step_05_spectral_features import extract_spectral_features


PILOT_OUTPUT = (
    RESULTS_DIR / "scientific_simulations" / "vnext_20261106"
    / "m2_georoc_labelled_pilot"
)


def _stable_sample_id(citation: str, source_id: str, lat: float, lon: float) -> str:
    key = f"GEOROC|{citation}|{source_id}|{lat:.6f}|{lon:.6f}"
    return "GEOROC_" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def collect_eligible_label_pairs(
    dataset: Path,
    existing_m2_manifest: Path,
    *,
    group_degrees: float = 0.5,
) -> pd.DataFrame:
    """Screen quality-B GEOROC samples with explicit source and coordinates.

    Li remains a supervision target and source information is audit metadata,
    not input to the spectral feature extractor.
    """
    if group_degrees <= 0:
        raise ValueError("Invalid spatial group size")
    original = pd.read_csv(dataset, low_memory=False)
    qc = protocol_masks(original)
    frame = original.loc[
        qc["B_reconstructed"]
        & original["source_dataset"].astype(str).eq("GEOROC")
        & original["Sample_type"].astype(str).eq("WHOLE ROCK")
    ].copy()
    required = ("source_sample", "CITATIONS", "Longitude", "Latitude",
                "Li_icpms", "Fe_total_provenance", "SUM (no water)")
    absent = [name for name in required if name not in frame.columns]
    if absent:
        raise ValueError(f"Missing provenance or Li columns: {absent}")

    frame["source_sample"] = frame["source_sample"].astype("string").str.strip()
    frame["CITATIONS"] = frame["CITATIONS"].astype("string").str.strip()
    for column in ("Longitude", "Latitude", "Li_icpms"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.loc[
        frame["source_sample"].notna() & frame["CITATIONS"].notna()
        & frame["source_sample"].ne("") & frame["CITATIONS"].ne("")
        & frame["Longitude"].between(-180, 180)
        & frame["Latitude"].between(-90, 90)
        & frame["Li_icpms"].ge(0)
    ].copy()
    reference = pd.read_csv(existing_m2_manifest)
    old_groups = set(reference["spatial_group"].astype(str))
    frame["spatial_group"] = [
        _spatial_group(lat, lon, group_degrees)
        for lat, lon in zip(frame["Latitude"], frame["Longitude"], strict=True)
    ]
    frame = frame.loc[~frame["spatial_group"].isin(old_groups)].copy()
    frame["sample_id"] = [
        _stable_sample_id(str(ref), str(name), lat, lon)
        for ref, name, lat, lon in zip(
            frame["CITATIONS"], frame["source_sample"],
            frame["Latitude"], frame["Longitude"], strict=True,
        )
    ]
    frame = frame.drop_duplicates(subset=["sample_id"])
    if frame.sample_id.duplicated().any():
        raise RuntimeError("Non-unique pilot sample identity")
    output = pd.DataFrame({
        "sample_id": frame["sample_id"],
        "Li_icpms": frame["Li_icpms"],
        "latitude": frame["Latitude"],
        "longitude": frame["Longitude"],
        "spatial_group": frame["spatial_group"],
        "source_citation": frame["CITATIONS"],
        "source_sample": frame["source_sample"],
        "iron_provenance": frame["Fe_total_provenance"],
        "analytical_sum_wt_pct": frame["SUM (no water)"],
        "coordinate_verification": "NOT_INDEPENDENTLY_VERIFIED",
        "training_eligibility": "TECHNICAL_PILOT_ONLY",
    })
    return output.reset_index(drop=True)


def select_small_pilot(
    candidates: pd.DataFrame, *, limit: int = 6, threshold_ppm: float = 21.6,
) -> pd.DataFrame:
    """At most one sample per new spatial group and publication.

    Select both sides of the descriptive lithium threshold with deterministic
    sorting. This is not a representative independent validation dataset.
    """
    if not 1 <= limit <= 20:
        raise ValueError("Pilot limit must be 1 to 20")
    required = {"sample_id", "spatial_group", "source_citation", "Li_icpms"}
    if not required.issubset(candidates.columns):
        raise ValueError("Missing sample metadata")
    frame = candidates.copy()
    frame["high_li_descriptive"] = frame["Li_icpms"].ge(threshold_ppm)
    frame["selection_sort"] = frame["sample_id"].astype(str).map(
        lambda x: hashlib.sha256(x.encode("utf-8")).hexdigest()
    )
    pools = {
        False: frame.loc[~frame.high_li_descriptive].sort_values("selection_sort"),
        True: frame.loc[frame.high_li_descriptive].sort_values("selection_sort"),
    }
    chosen = []
    groups = set()
    papers = set()
    while len(chosen) < limit:
        found = False
        # Alternate low and high Li to avoid class-only technical pilot.
        for high in (False, True):
            for index, row in pools[high].iterrows():
                if index in chosen or row.spatial_group in groups:
                    continue
                if row.source_citation in papers:
                    continue
                chosen.append(index)
                groups.add(row.spatial_group)
                papers.add(row.source_citation)
                found = True
                break
            if len(chosen) >= limit:
                break
        if not found:
            break
    out = frame.loc[chosen].drop(columns="selection_sort").reset_index(drop=True)
    if out.spatial_group.duplicated().any():
        raise RuntimeError("Pilot reused a spatial group")
    if out.source_citation.duplicated().any():
        raise RuntimeError("Pilot reused a publication")
    return out


def _quality_of_patch(image: np.ndarray, bands: tuple[str, ...]) -> dict:
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[0] != len(bands):
        raise ValueError(f"Sentinel shape incompatible: {image.shape}")
    finite = np.isfinite(image)
    share = finite.mean(axis=(1, 2))
    return {
        "shape": [int(x) for x in image.shape],
        "finite_per_band": {name: float(f) for name, f in zip(bands, share)},
        "technically_usable": bool(np.all(share >= 0.8)),
    }


def run_pilot(
    root: Path = PROJECT_ROOT, output: Path = PILOT_OUTPUT,
    *, limit: int = 6, acquire: bool = False,
) -> dict:
    """Generate deterministic pilot and optionally download up to limit patches.

    No dataset update, no training, no release, no cloud paid services.
    """
    config = load_config("model_2")
    frame = collect_eligible_label_pairs(
        root / "results/scientific_simulations/vnext_20261106"
        / "stage2_m1/feot_combined_experiment_only.csv",
        root / "data/processed/model_2/training_manifest.csv",
        group_degrees=float(config["training"]["spatial_group_degrees"]),
    )
    reference = pd.read_csv(root / "data/processed/model_2/training_manifest.csv")
    threshold = float(reference["Li_icpms"].quantile(0.75))
    selected = select_small_pilot(frame, limit=limit, threshold_ppm=threshold)
    output.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output / "selected_candidates.csv", index=False)
    selected.to_json(
        output / "selected_candidates.json", orient="records", indent=2,
        force_ascii=False,
    )
    rows = []
    features = []
    provider = None
    provider_failure = None
    if acquire and len(selected):
        provider = Sentinel2Provider(_sentinel_config(config))
        try:
            provider.validate()
        except SentinelProviderError as exc:
            provider_failure = str(exc)
            provider = None
    for sample in selected.itertuples(index=False):
        record = {
            "sample_id": sample.sample_id,
            "spatial_group": sample.spatial_group,
            "Li_icpms": float(sample.Li_icpms),
            "source_citation": sample.source_citation,
            "coordinate_verification": "NOT_INDEPENDENTLY_VERIFIED",
            "status": "SELECTED_NO_DOWNLOAD",
        }
        if acquire:
            if provider is None:
                record["status"] = "PROVIDER_UNAVAILABLE"
                record["error"] = provider_failure
            else:
                try:
                    patch, metadata = download_patch(
                        sample_id=sample.sample_id,
                        latitude=float(sample.latitude),
                        longitude=float(sample.longitude),
                        destination=output/"patches"/(sample.sample_id+".npy"),
                        provider=provider,
                    )
                    image = load_imagery(patch)
                    qc = _quality_of_patch(image, provider.config.bands)
                    record.update({
                        "status": "PATCH_VALID" if qc["technically_usable"]
                                  else "PATCH_INCOMPLETE",
                        "image_path": str(patch),
                        "scene_id": metadata.get("scene_id"),
                        "scene_datetime": metadata.get("datetime"),
                        "cloud_cover": metadata.get("cloud_cover"),
                        "patch_quality": qc,
                    })
                    if qc["technically_usable"]:
                        processed = preprocess_image(
                            image,
                            normalize_per_band=bool(
                                config["imagery"].get("normalize_per_band", False)
                            ),
                        )
                        vector = extract_spectral_features(
                            processed, band_names=provider.config.bands,
                        )
                        # Only spectral output is used as model input.
                        features.append({"sample_id": sample.sample_id, **vector})
                except (SentinelProviderError, SentinelSceneUnavailable,
                        SentinelAssetError, ValueError, OSError) as exc:
                    record["status"] = "PATCH_FAILED"
                    record["error"] = str(exc)
        rows.append(record)
        print("PILOT_SAMPLE",sample.sample_id,record["status"],flush=True)
    report = {
        "source": "GEOROC analytical Li used only as sample-level supervised label",
        "candidate_total_excluding_existing_geographical_groups": len(frame),
        "selected": len(selected),
        "download_requested": bool(acquire),
        "download_limit": limit,
        "patch_valid": sum(x["status"] == "PATCH_VALID" for x in rows),
        "patch_failed": sum(x["status"] == "PATCH_FAILED" for x in rows),
        "provider_error": provider_failure,
        "data_usage": "Experimental patch cache only. No canonical changes.",
        "model_input": "Sentinel-2 spectral features only, unchanged 6-band extractor",
        "labels_certified_for_final_training": False,
        "source_accuracy_caution": "Original GEOROC coordinates still require source-level independent verification.",
        "individual_results": rows,
    }
    if features:
        pd.DataFrame(features).to_csv(
            output / "pilot_spectral_features.csv", index=False
        )
        report["feature_columns"] = len(features[0]) - 1
    (output / "pilot_manifest.json").write_text(
        json.dumps(report,indent=2,ensure_ascii=False,default=str),encoding="utf8"
    )
    print("PILOT_SUMMARY",json.dumps({
        key: value for key, value in report.items()
        if key not in {"individual_results", "source"}
    }, ensure_ascii=False),flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit",type=int,default=6)
    parser.add_argument("--acquire",action="store_true")
    args = parser.parse_args()
    run_pilot(limit=args.limit,acquire=args.acquire)
