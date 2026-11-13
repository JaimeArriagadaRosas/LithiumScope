"""Explicit, provenance-preserving GEOROC site prefet ch for M2 experiments.

Selected samples remain PROVISIONAL; they are not automatically included
in any trained M2 run, nor are source concentration values changed.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from lithiumscope.core.config import load_config
from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.paths import PROJECT_ROOT
from lithiumscope.model_2.data.builder import _sentinel_config
from lithiumscope.model_2.data.sentinel2 import (
    Sentinel2Provider,
    SentinelAssetError,
    SentinelSceneUnavailable,
)
from lithiumscope.model_2.data.sentinel2_patch_io import download_patch
from lithiumscope.model_2.methodology.rededge_acquisition import (
    cached_rededge_sample,
)
from lithiumscope.model_2.methodology.satellite_quality import (
    ScientificScenePolicy,
)

OUT = PROJECT_ROOT / "results/scientific_simulations/m2_rededge_v4_20261113/new_sites"
INVENTORY = PROJECT_ROOT / (
    "results/scientific_simulations/m2_spectral_expansion_20261113/georoc_candidate_inventory.csv"
)
MANIFEST = PROJECT_ROOT / ("data/processed/model_2/vnext_expanded/training_manifest.csv")
TIME_WINDOW = "2025-01-01/2026-11-12"


def plan(*, output: Path = OUT) -> pd.DataFrame:
    inventory = pd.read_csv(INVENTORY)
    old = pd.read_csv(MANIFEST)
    current_groups = set(old.spatial_group.astype(str))
    if not inventory.sample_id.is_unique or len(old) != 783:
        raise ValueError("M2 new-site input identity changed")
    # Only sites in genuinely not-yet-represented spatial groups.
    pool = inventory.loc[
        inventory.next_action.isin(
            [
                "REVIEW_23_PREVIOUS_DOWNLOAD_FAILURES",
                "NEW_GROUP_NEEDS_IMAGE_AND_SOURCE_QA",
            ]
        )
        & ~inventory.spatial_group.isin(current_groups)
    ].copy()
    if not pool.sample_id.is_unique:
        raise ValueError("Repeated source identifiers")
    # One provisional sample per 0.5° group, balanced by source confidence.
    pool["is_previously_queued"] = pool.next_action.eq(
        "REVIEW_23_PREVIOUS_DOWNLOAD_FAILURES",
    )
    pool["has_rock_class"] = ~pool.raw_rock_class.fillna("").str.upper().str.contains(
        "NOT GIVEN|UNKNOWN",
    )
    pool = pool.sort_values(
        ["is_previously_queued", "has_rock_class", "site_count", "sample_id"],
        ascending=[False, False, True, True],
    )
    selected = pool.drop_duplicates("spatial_group").reset_index(drop=True)
    if selected.spatial_group.duplicated().any():
        raise ValueError("New site queue repeats a spatial group")
    selected["source_verification"] = "PROVISIONAL_NOT_FIELD_VERIFIED"
    output.mkdir(parents=True, exist_ok=True)
    meta = {
        "version": "m2_provisional_new_sites_v4_20261113",
        "source_inventory_sha256": file_sha256(INVENTORY),
        "original_783_manifest_sha256": file_sha256(MANIFEST),
        "n_sites": len(selected),
        "n_new_groups": int(selected.spatial_group.nunique()),
        "source_labels_provisional": True,
        "no_automatic_training": True,
        "datetime_window": TIME_WINDOW,
        "six_original_bands_then_four_real_rededge_bands": True,
    }
    contract = output / "site_plan_contract.json"
    queue = output / "new_site_queue.csv"
    if contract.is_file():
        if json.loads(contract.read_text(encoding="utf-8")) != meta:
            raise ValueError("Source queue drift; do not overwrite old plan")
        previous = pd.read_csv(queue)
        if not selected.equals(previous[selected.columns]):
            raise ValueError("New-site queue changed")
    else:
        selected.to_csv(queue, index=False)
        contract.write_text(
            json.dumps(meta, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    print("M2_V4_NEW_GROUP_PLAN", meta, flush=True)
    return selected


def fetch(*, max_sites: int, output: Path = OUT, retry_failed: bool = False) -> dict:
    if max_sites < 1:
        raise ValueError("max_sites must be positive")
    queue = plan(output=output)
    # Prefer genuinely NEW 20?30 S groups; retry previous failure only by opt-in.
    queue = queue.assign(
        priority_region=queue.Latitude.between(-30, -20),
        priority_new=queue.next_action.eq("NEW_GROUP_NEEDS_IMAGE_AND_SOURCE_QA"),
    ).sort_values(
        ["priority_region", "priority_new"],
        ascending=[False, False],
        kind="stable",
    )
    cfg = load_config("model_2")
    config = replace(_sentinel_config(cfg), cloud_cover_max=25.0)
    if tuple(config.bands) != (
        "red",
        "green",
        "blue",
        "nir",
        "swir16",
        "swir22",
    ):
        raise ValueError("Frozen M2 original six-band schema changed")
    provider = Sentinel2Provider(config)
    policy = ScientificScenePolicy(
        min_joint_valid=0.95,
        max_scl_blocked=0.25,
        min_not_vegetated=0.50,
        max_scene_cloud=25,
        max_candidates=20,
    )
    old = output / "source_spectra_accepted.csv"
    history = pd.read_csv(old) if old.exists() else pd.DataFrame()
    done = set(history.sample_id.astype(str)) if not history.empty else set()
    failure_path = output / "site_acquisition_failures.csv"
    failures = pd.read_csv(failure_path) if failure_path.exists() else pd.DataFrame()
    blocked = set() if retry_failed or failures.empty else set(failures.sample_id.astype(str))
    attempted = 0
    success = 0
    for candidate in queue.itertuples(index=False):
        if str(candidate.sample_id) in done or str(candidate.sample_id) in blocked:
            continue
        if attempted >= max_sites:
            break
        attempted += 1
        sample_id = str(candidate.sample_id)
        destination = output / "original_6_band_patches" / f"{sample_id}.npy"
        try:
            scene, qa, _ = provider.search_validated_scene(
                float(candidate.Latitude),
                float(candidate.Longitude),
                datetime=TIME_WINDOW,
                max_candidates=20,
                policy=policy,
            )
            path, meta = download_patch(
                sample_id=sample_id,
                latitude=float(candidate.Latitude),
                longitude=float(candidate.Longitude),
                destination=destination,
                provider=provider,
                selected_scene=scene,
                scientific_quality=qa,
            )
            if not qa.get("scientifically_accepted_proxy"):
                raise ValueError("Scene failed local six-band SCL QA")
            _, re_meta = cached_rededge_sample(
                SimpleNamespace(
                    sample_id=sample_id,
                    image_path=str(path),
                    sentinel_scene_id=meta["scene_id"],
                    latitude=candidate.Latitude,
                    longitude=candidate.Longitude,
                ),
                output=output / "rededge_4_band_patches",
                min_joint_valid=0.95,
            )
            record = {
                "sample_id": sample_id,
                "Li_icpms": float(candidate.Li_icpms),
                "latitude": float(candidate.Latitude),
                "longitude": float(candidate.Longitude),
                "spatial_group": str(candidate.spatial_group),
                "image_path": str(path),
                "sentinel_scene_id": meta["scene_id"],
                "source_doi": str(candidate.source_dois),
                "source_sample": str(candidate.source_sample),
                "raw_rock_class": str(candidate.raw_rock_class),
                "source_label_status": "PROVISIONAL_NO_FIELD_VALIDATION",
                "scientific_proxy_accepted": True,
                "rededge_joint_valid_fraction": re_meta["rededge_joint_valid_fraction"],
                "original_patch_sha256": re_meta["original_patch_sha256"],
                "rededge_patch_sha256": re_meta["rededge_patch_sha256"],
            }
            history = pd.concat([history, pd.DataFrame([record])], ignore_index=True)
            history.to_csv(old, index=False)
            done.add(sample_id)
            success += 1
            print("M2_V4_SITE_ACCEPTED", sample_id, "scene", meta["scene_id"], flush=True)
        except Exception as exc:
            fail = {
                "sample_id": sample_id,
                "failure_type": type(exc).__name__,
                "reason": str(exc)[:250],
            }
            failures = pd.concat([failures, pd.DataFrame([fail])], ignore_index=True)
            failures.to_csv(failure_path, index=False)
            print("M2_V4_SITE_REJECTED", sample_id, str(exc)[:140], flush=True)
    result = {
        "attempted": attempted,
        "newly_accepted": success,
        "total_accepted": len(history),
        "all_labels_remain_provisional": True,
        "active_models_not_changed": True,
    }
    (output / "prefetch_status.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fetch", type=int, default=0)
    p.add_argument("--retry-failed", action="store_true")
    args = p.parse_args()
    if args.fetch:
        print("M2_V4_FETCH_STATUS", fetch(max_sites=args.fetch, retry_failed=args.retry_failed), flush=True)
    else:
        plan()


if __name__ == "__main__":
    main()
