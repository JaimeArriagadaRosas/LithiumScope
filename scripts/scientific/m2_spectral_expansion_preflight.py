"""Read-only input audit for M2 spectral and new-site expansion.

The output contains candidate GEOROC records, NOT new training labels or
spectral observations. No image downloads or active model changes.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lithiumscope.core.paths import PROJECT_ROOT


def groups(lat: pd.Series, lon: pd.Series) -> pd.Series:
    a = np.floor((lat.to_numpy(float) + 90.0) / 0.5).astype(int)
    b = np.floor((lon.to_numpy(float) + 180.0) / 0.5).astype(int)
    return pd.Series([f"g_{x}_{y}" for x, y in zip(a, b)], index=lat.index)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(root: Path = PROJECT_ROOT, *, output: Path | None = None) -> dict:
    source = root / (
        "results/scientific_simulations/vnext_20261106/"
        "georoc_source_validation/georoc_B_screened_sites.csv"
    )
    queue_path = root / "data/interim/model_2/vnext_georoc/selected_candidate_queue.csv"
    expanded_path = root / "data/processed/model_2/vnext_expanded/training_manifest.csv"
    pool = pd.read_csv(source)
    queue = pd.read_csv(queue_path)
    expanded = pd.read_csv(expanded_path)
    assert len(expanded) == 783 and len(queue) == 100
    assert expanded.sample_id.is_unique and queue.sample_id.is_unique
    historical = expanded.iloc[:706]
    assert historical.source_dataset.eq("MAMANI09_HISTORICAL").all()
    selected = set(queue.sample_id.astype(str))
    downloaded = set(expanded.sample_id.astype(str))
    existing_groups = set(expanded.spatial_group.astype(str))
    eligible = pool.loc[
        pool.screening_tier.eq("PROVISIONAL_SITE_REVIEW")
        & pool.references_complete.eq(True)
        & pool.Li_icpms.le(100.6)
        & pool.Latitude.between(-90, 90)
        & pool.Longitude.between(-180, 180)
    ].copy()
    assert eligible.sample_id.is_unique
    eligible["spatial_group"] = groups(eligible.Latitude, eligible.Longitude)
    eligible["queued"] = eligible.sample_id.astype(str).isin(selected)
    eligible["downloaded"] = eligible.sample_id.astype(str).isin(downloaded)
    eligible["group_already_present"] = eligible.spatial_group.isin(existing_groups)
    if (eligible.downloaded & ~eligible.queued).any():
        raise ValueError("Downloaded candidate absent from original queue")
    eligible["next_action"] = np.select(
        [
            eligible.downloaded,
            eligible.queued,
            ~eligible.group_already_present,
        ],
        [
            "EXISTING_77_NO_NEW_LABEL",
            "REVIEW_23_PREVIOUS_DOWNLOAD_FAILURES",
            "NEW_GROUP_NEEDS_IMAGE_AND_SOURCE_QA",
        ],
        default="EXISTING_SPATIAL_GROUP_NO_INDEPENDENT_COUNT",
    )
    eligible = eligible.sort_values(
        ["next_action", "spatial_group", "sample_id"]
    ).reset_index(drop=True)
    stats = {
        "status": "source_inventory_only_no_download_no_model_change",
        "source_rows": int(len(pool)),
        "source_provisional": int(pool.screening_tier.eq("PROVISIONAL_SITE_REVIEW").sum()),
        "eligible_after_source_filter": int(len(eligible)),
        "prior_queue": int(len(queue)),
        "prior_77_downloaded": int((eligible.downloaded).sum()),
        "queued_not_downloaded": int((eligible.queued & ~eligible.downloaded).sum()),
        "unqueued_candidate_rows": int((~eligible.queued).sum()),
        "unqueued_new_group_count": int(
            eligible.loc[
                ~eligible.queued & ~eligible.group_already_present, "spatial_group"
            ].nunique()
        ),
        "historical_rows": int(len(historical)),
        "current_expanded_rows": int(len(expanded)),
        "group_count_existing": int(expanded.spatial_group.nunique()),
        "group_count_eligible_candidates": int(eligible.spatial_group.nunique()),
        "source_sha256": sha(source),
        "queue_sha256": sha(queue_path),
        "current_manifest_sha256": sha(expanded_path),
        "cautions": [
            "Not an additional ground truth count",
            "Existing groups are not independent new spatial tests",
            "Rock-to-image match and spectral quality must be verified",
            "Do not merge these candidates into active training automatically",
        ],
    }
    if output is None:
        output = root / "results/scientific_simulations/m2_spectral_expansion_20261113"
    output.mkdir(parents=True, exist_ok=True)
    eligible.to_csv(output / "georoc_candidate_inventory.csv", index=False)
    (output / "candidate_inventory_summary.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("M2_SPECTRAL_SOURCE_PREFLIGHT", json.dumps(stats), flush=True)
    return stats


if __name__ == "__main__":
    run()
