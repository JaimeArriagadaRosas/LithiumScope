"""Multi-date within-season comparisons, audit scene overlap and satellite mixture."""
from pathlib import Path
import hashlib
import json

import pandas as pd


def analyze(root: Path) -> Path:
    base = root / "results" / "scientific_simulations"
    out = base / "temporal_multiscene" / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    sources = [
        ("existing_2024", base / "seasonal_2024/seasonal_manifest.csv",
         base / "seasonal_2024/batch/batch_m2.csv"),
        ("existing_2025", base / "campaign_2025/campaign_manifest.csv",
         base / "campaign_2025/batch/batch_m2.csv"),
        ("additional", base / "temporal_multiscene/additional_manifest.csv",
         base / "temporal_multiscene/batch/batch_m2.csv"),
    ]
    records, hashes = [], {}
    for label, manifest_file, metric_file in sources:
        meta = pd.read_csv(manifest_file)
        metrics = pd.read_csv(metric_file)
        for path in (manifest_file, metric_file):
            hashes[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
        if label == "existing_2025":
            meta = meta.loc[meta.image_id.str.contains(r"_(?:summer|winter)_", regex=True)].copy()
        baseline = metrics.loc[(metrics.scenario == "baseline") & (metrics.status == "SCORED"),
                               ["image_id", "score", "ood_fraction"]]
        rows = meta.merge(baseline, on="image_id", how="inner", validate="one_to_one")
        rows["campaign"] = label
        rows["region"] = rows.image_id.str.split("_").str[0]
        rows["season"] = rows.image_id.str.extract(r"_(summer|winter)")[0]
        rows["satellite"] = rows.scene_id.str.split("_").str[0]
        records.append(rows)
    combined = pd.concat(records, ignore_index=True)
    combined["year"] = combined.year.astype(int)
    combined["scene_key"] = combined["region"] + "|" + combined["scene_id"]
    combined["duplicate_scene_region"] = combined.duplicated("scene_key", keep=False)
    combined.to_csv(out / "scene_audit.csv", index=False)
    distinct = combined.drop_duplicates("scene_key")
    grouped = distinct.groupby(["region", "year", "season"]).agg(
        scene_count=("scene_id", "nunique"),
        satellite_count=("satellite", "nunique"),
        mean_score=("score", "mean"),
        std_score=("score", lambda v: v.std(ddof=0)),
        min_score=("score", "min"),
        max_score=("score", "max"),
    ).reset_index()
    grouped["score_range"] = grouped.max_score - grouped.min_score
    grouped.to_csv(out / "within_season_variability.csv", index=False)
    paired = grouped.pivot(index=["region", "year"], columns="season", values="mean_score")
    paired.columns = ["mean_" + col for col in paired.columns]
    paired = paired.reset_index()
    paired["winter_minus_summer"] = paired.mean_winter - paired.mean_summer
    paired.to_csv(out / "seasonal_mean_comparisons.csv", index=False)
    output = {
        "all_observations": len(combined),
        "unique_region_scene": len(distinct),
        "duplicate_scene_region_rows": int(combined.duplicate_scene_region.sum()),
        "groups": len(grouped),
        "groups_with_two_or_more_scenes": int((grouped.scene_count >= 2).sum()),
        "sources_sha256": hashes,
        "cautions": [
            "Two captures per season remain small and can represent different atmospheric states.",
            "Acquisition dates, sensors and reflectance distributions may be confounded.",
            "Scores do not validate lithium labels; no external accuracy measurements are inferred.",
        ],
    }
    (out / "analysis_manifest.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    print(analyze(Path.cwd()))

