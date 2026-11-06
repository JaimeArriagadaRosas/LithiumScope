"""Evidence-gated catalog for independent geochemistry, without fabricating observations."""
from __future__ import annotations
from pathlib import Path
import json
import pandas as pd

REQUIRED = ("sample_id","material","latitude","longitude","Li_value",
            "Li_unit","analytical_method","source","sample_date","independent_of_training")
COMPATIBLE_MATERIAL = {"rock", "whole_rock"}


def validate_geochemistry(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing = [col for col in REQUIRED if col not in frame]
    if missing:
        raise ValueError("Missing required evidence columns: " + ", ".join(missing))
    table = frame.copy()
    numeric = pd.to_numeric(table["Li_value"], errors="coerce")
    lat = pd.to_numeric(table["latitude"], errors="coerce")
    lon = pd.to_numeric(table["longitude"], errors="coerce")
    independent = table["independent_of_training"].astype(str).str.lower().isin(["true","yes","1"])
    compatible = (
        table["material"].astype(str).str.lower().isin(COMPATIBLE_MATERIAL)
        & table["Li_unit"].astype(str).str.lower().isin(["ppm", "mg/kg"])
        & numeric.ge(0) & numeric.notna()
        & lat.between(-90,90) & lon.between(-180,180)
        & independent
        & table["source"].notna() & table["analytical_method"].notna()
        & table["sample_id"].notna()
    )
    table["evidence_status"] = compatible.map({True:"CANDIDATE_FOR_REVIEW",False:"NOT_ELIGIBLE"})
    return table.loc[compatible].copy(), table.loc[~compatible].copy()


def initialize_catalog(directory: Path) -> Path:
    directory.mkdir(parents=True,exist_ok=True)
    template = directory/"independent_geochemistry_template.csv"
    if not template.exists():
        pd.DataFrame(columns=REQUIRED).to_csv(template,index=False)
    (directory/"README.md").write_text(
        "# Independent geochemical evidence\n\n"
        "Populate the template with measured geochemistry and traceable source, method and unit. "
        "Rock measurements and brine must remain separate; material 'brine' is not eligible "
        "for M1 validation. Check geographic distance and sample overlap against training "
        "before declaring the candidate truly external. No synthetic values are accepted "
        "as ground truth.\n",encoding="utf-8")
    return template


def review_catalog(source: Path, destination: Path) -> Path:
    eligible, excluded = validate_geochemistry(pd.read_csv(source))
    destination.mkdir(parents=True,exist_ok=True)
    eligible.to_csv(destination/"review_candidates.csv",index=False)
    excluded.to_csv(destination/"excluded.csv",index=False)
    report=destination/"evidence_quality.json"
    report.write_text(json.dumps({"candidates":len(eligible),"excluded":len(excluded),
        "note":"Candidates require explicit leakage/overlap audit prior to use as external ground truth."},
        indent=2),encoding="utf-8")
    return report
