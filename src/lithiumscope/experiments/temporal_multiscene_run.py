"""Additional within-season Sentinel-2 dates, distinct from prior March/August samples.

Read-only inference campaign. The windows are deliberately different from prior
seasonal acquisitions. All output stays in an ignored experiment directory.
"""
from pathlib import Path

import pandas as pd

from lithiumscope.experiments.acquisition import PILOT_LOCATIONS, acquire_pilot
from lithiumscope.experiments.batch import run_m2_batch, summarize
from lithiumscope.persistence.load_model import load_latest_model

DEST = Path("results/scientific_simulations/temporal_multiscene")
WINDOWS = {
    "summer_early": ("01-01", "02-15"),
    "winter_early": ("06-01", "07-15"),
}


def main() -> None:
    inputs = []
    for year in (2025, 2024):
        for phase, dates in WINDOWS.items():
            positions = {f"{area}_{phase}": value for area, value in PILOT_LOCATIONS.items()}
            folder = DEST / str(year) / phase
            manifest, _ = acquire_pilot(
                folder, years=(year,), locations=positions,
                date_window=dates, max_patches=len(positions), timeout_seconds=45,
            )
            data = pd.read_csv(manifest)
            data["phase"] = phase
            data["year"] = year
            inputs.append(data)
            print("PHASE", year, phase, data.status.value_counts().to_dict(), flush=True)
    combined = DEST / "additional_manifest.csv"
    DEST.mkdir(parents=True, exist_ok=True)
    pd.concat(inputs, ignore_index=True).to_csv(combined, index=False)
    model, _ = load_latest_model("model_2")
    result = run_m2_batch(model, combined, DEST / "batch")
    print("BATCH", result, flush=True)
    print("SUMMARY", summarize(result, result.parent), flush=True)


if __name__ == "__main__":
    main()
