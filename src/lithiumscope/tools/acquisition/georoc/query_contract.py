from __future__ import annotations

import json
from pathlib import Path

GEOROC_QUERY_URL = (
    "https://georoc.eu/georoc/Chemistry.asp"
)
GEOROC_FILTERED_NAME = (
    "GEOROC_Andean_Arc_LithiumScope.csv"
)

CHEMISTRY = (
    "LI",
    "SIO2",
    "TIO2",
    "AL2O3",
    "FE2O3",
    "FE2O3T",
    "FEOT",
    "MNO",
    "MGO",
    "CAO",
    "NA2O",
    "K2O",
    "P2O5",
    "TH",
    "U",
    "RB",
    "CS",
    "NB",
    "TA",
    "PB",
    "BA",
    "SR",
    "ZR",
    "V",
    "HF",
)


def acquisition_contract() -> dict:
    return {
        "query_url": GEOROC_QUERY_URL,
        "scope": "ANDEAN ARC",
        "material": "WHOLE ROCK",
        "chemistry": list(CHEMISTRY),
        "destination_name": (
            GEOROC_FILTERED_NAME
        ),
        "massive_precompiled_fallback": False,
    }


def write_acquisition_contract(
    path: Path,
) -> Path:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    path.write_text(
        json.dumps(
            acquisition_contract(),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path
