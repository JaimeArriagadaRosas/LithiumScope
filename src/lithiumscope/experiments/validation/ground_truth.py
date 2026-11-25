"""Evidence-gated catalog for independent geochemistry, without fabricating observations."""
from __future__ import annotations
from pathlib import Path
from lithiumscope.results.atomic_exports import write_json, write_table
import pandas as pd

from lithiumscope.datasets.validation.external_geochemistry import REQUIRED, COMPATIBLE_MATERIAL, validate_geochemistry


def initialize_catalog(directory: Path) -> Path:
    directory.mkdir(parents=True,exist_ok=True)
    template = directory/"independent_geochemistry_template.csv"
    if not template.exists():
        write_table(template,pd.DataFrame(columns=REQUIRED))
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
    write_table(destination/"review_candidates.csv",eligible)
    write_table(destination/"excluded.csv",excluded)
    report=destination/"evidence_quality.json"
    write_json(report,{"candidates":len(eligible),"excluded":len(excluded),
        "note":"Candidates require explicit leakage/overlap audit prior to use as external ground truth."},
        )
    return report
