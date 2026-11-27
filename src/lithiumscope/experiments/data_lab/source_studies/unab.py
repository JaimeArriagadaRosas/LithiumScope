"""Read-only audit for a public candidate dataset with explicit Li exclusion."""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
from lithiumscope.experiments.data_lab.source_studies.georoc.protocol import EXPERIMENT_ROOT
import hashlib
import io
import json

import pandas as pd
import requests

API = "https://repositoriodedatos.unab.cl/api"
DOI = "doi:10.60483/UNAB/PLCJLQ"
TABLE_ID = 592


def audit(destination: Path, *, acquire=False):
    if not acquire:
        return {"state": "preflight", "doi": DOI}
    destination.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    metadata_response = session.get(
        f"{API}/datasets/:persistentId/", params={"persistentId": DOI}, timeout=30)
    metadata_response.raise_for_status()
    metadata = metadata_response.json()["data"]
    original = session.get(f"{API}/access/datafile/{TABLE_ID}", timeout=45)
    original.raise_for_status()
    table = pd.read_csv(io.BytesIO(original.content), sep="\t", low_memory=False)
    first_column = table.iloc[:, 0].astype(str).str.strip()
    # This dataset stores chemical analyte names as rows, not columns.
    has_li = first_column.str.fullmatch(r"Li|Lithium|Li\s*\(ppm\)", case=False).any()
    has_coordinates = {"Latitude", "Longitude"}.issubset(set(first_column))
    result = {
        "source": metadata.get("persistentUrl"),
        "accessed_utc": datetime.now(timezone.utc).isoformat(),
        "raw_sha256": hashlib.sha256(original.content).hexdigest(),
        "source_filename": "Table_S01-Chemical&SrNd_Analyses.tab",
        "shape_as_published": list(table.shape),
        "has_coordinates": bool(has_coordinates),
        "has_lithium_analyte": bool(has_li),
        "sample_columns": int(table.shape[1] - 1),
        "status": "EXCLUDED_NO_LITHIUM_LABEL" if not has_li else "REQUIRES_FURTHER_REVIEW",
        "independent_model_2_labels_accepted": 0,
        "note": "External chemistry alone does not validate lithium prediction without measured Li and source-overlap audit.",
    }
    target = destination / "unab_chemistry_review.json"
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    return target

if __name__=="__main__":
    from lithiumscope.core.logging.session import run_log
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquire",action="store_true")
    if not parser.parse_args().acquire:
        print("Preflight: use --acquire to authorize network requests.")
        raise SystemExit(0)
    with run_log("simulations","unab_source_review"):
        audit(EXPERIMENT_ROOT, acquire=True)
