"""Inspect existing Mamani09+GEOROC samples; no downloader, no training."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from lithiumscope.core.config import load_config
from lithiumscope.core.paths import PROJECT_ROOT, RESULTS_DIR
from lithiumscope.datasets.harmonization import canonicalize_coordinates
from lithiumscope.experiments.analysis.georoc_qc_protocols import (
    protocol_masks,
    protocol_summary,
)
from lithiumscope.model_1.steps.step_02_detection_limits import clean_detection_limits
from lithiumscope.model_1.steps.step_03_missing_values import normalize_missing_values
from lithiumscope.model_1.steps.step_05_target_filtering import filter_target


def run_stage1(
    root: Path = PROJECT_ROOT, destination: Path | None = None,
) -> dict:
    destination = destination or (
        RESULTS_DIR / "scientific_simulations" / "vnext_20261106" / "stage1"
    )
    config = load_config("model_1")["data"]
    source = root / "data/processed/model_1/training_combined.csv"
    if not source.is_file():
        raise FileNotFoundError(source)
    frame = pd.read_csv(source, low_memory=False)
    frame = canonicalize_coordinates(
        normalize_missing_values(clean_detection_limits(frame))
    )
    valid, _ = filter_target(frame, list(config["target_candidates"]))
    masks = protocol_masks(
        valid, lower=float(config["quality_min"]),
        upper=float(config["quality_max"]),
    )
    summary = protocol_summary(valid, masks)
    destination.mkdir(parents=True, exist_ok=True)
    summary.to_csv(destination / "qc_protocol_summary.csv", index=False)
    audit = {
        "dataset": str(source),
        "rows_loaded": int(len(frame)),
        "rows_with_valid_li": int(len(valid)),
        "protocol_counts": summary.to_dict("records"),
        "stages": "A=current; B=ten-oxide complete reconstructed; C=unverified experimental",
        "promote_C_to_production": False,
    }
    raw = root / "data/raw/model_1/georoc/GEOROC_Andean_Arc_LithiumScope.csv"
    if raw.is_file():
        rock = pd.read_csv(raw, low_memory=False)
        li = pd.to_numeric(rock.get("LI(PPM)"), errors="coerce")
        raw_summary = {
            "rows": len(rock),
            "lithium_numeric": int(li.notna().sum()),
        }
        for column in (
            "FE2O3(WT%)", "FE2O3T(WT%)", "FEO(WT%)", "FEOT(WT%)",
            "SIO2(WT%)", "AL2O3(WT%)",
        ):
            if column in rock.columns:
                v = pd.to_numeric(rock[column], errors="coerce")
                raw_summary[column] = {
                    "all": int(v.notna().sum()),
                    "with_li": int((v.notna() & li.notna()).sum()),
                }
        audit["raw_georoc_iron_coverage"] = raw_summary
    geo = valid.loc[valid["source_dataset"].astype(str).eq("GEOROC")]
    geo_mask = masks.loc[geo.index]
    c_extra = geo_mask.loc[geo_mask["C_unverified_extra"]]
    audit["C_provisional_extra_georoc"] = len(c_extra)
    audit["C_extra_lithium_quantiles"] = (
        {str(k): float(v) for k, v in pd.to_numeric(
            geo.loc[c_extra.index, "Li_icpms"], errors="coerce"
        ).quantile([0.1, 0.5, 0.9]).items()}
        if len(c_extra) else {}
    )
    (destination / "stage1_evidence.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return audit


if __name__ == "__main__":
    result = run_stage1()
    print(json.dumps(result, indent=2, ensure_ascii=False))
