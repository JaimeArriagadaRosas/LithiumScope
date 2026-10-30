"""Summarize inter-location and inter-year changes for accepted unlabelled patches."""
from pathlib import Path
import json

import pandas as pd


def compare_geographies(batch_file: Path, destination: Path) -> tuple[Path, Path]:
    data = pd.read_csv(batch_file)
    destination.mkdir(parents=True, exist_ok=True)
    baseline = data.loc[
        (data["scenario"] == "baseline") & (data["status"] == "SCORED"),
        ["image_id", "scene_id", "year", "score", "ood_fraction"],
    ].copy()
    baseline["area"] = baseline["image_id"].map(lambda item: str(item).rsplit("_", 1)[0])
    baseline = baseline.sort_values(["area", "year"])
    baseline["year_to_year_shift"] = baseline.groupby("area")["score"].diff()
    output = destination / "geography_comparison.csv"
    baseline.to_csv(output, index=False)
    stats = {
        "accepted_image_count": len(baseline),
        "areas": int(baseline["area"].nunique()),
        "years": sorted(int(x) for x in baseline["year"].dropna().unique()),
        "score_min": float(baseline["score"].min()) if len(baseline) else None,
        "score_max": float(baseline["score"].max()) if len(baseline) else None,
        "interpretation": "Scores represent relative spectral prioritization; no lithium ground truth.",
    }
    report = destination / "geography_comparison.json"
    report.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    return output, report
