from pathlib import Path

import pandas as pd

from lithiumscope.datasets.georoc_ingestion import (
    ingest_georoc_files,
)


def test_georoc_ingestion_maps_core_fields(
    tmp_path: Path,
):
    source = tmp_path / "georoc.csv"
    pd.DataFrame(
        [
            {
                "SAMPLE NAME": "G-1",
                "TYPE OF MATERIAL": "WHOLE ROCK",
                "LONGITUDE": -70.1,
                "LATITUDE": -23.4,
                "LI(PPM)": 15.0,
                "SIO2(WT%)": 60.0,
                "TIO2(WT%)": 1.0,
                "AL2O3(WT%)": 16.0,
                "FE2O3T(WT%)": 6.0,
                "MNO(WT%)": 0.1,
                "MGO(WT%)": 4.0,
                "CAO(WT%)": 6.0,
                "NA2O(WT%)": 3.0,
                "K2O(WT%)": 3.0,
                "P2O5(WT%)": 0.2,
            }
        ]
    ).to_csv(source, index=False)

    destination = tmp_path / "harmonized.csv"
    result = ingest_georoc_files(
        [source],
        destination,
        chunksize=1,
        minimum_predictors=8,
    )

    frame = pd.read_csv(result.path)
    assert result.rows_kept == 1
    assert frame.loc[0, "Li_icpms"] == 15.0
    assert frame.loc[0, "Longitude"] == -70.1
    assert frame.loc[0, "source_dataset"] == "GEOROC"
    assert frame.loc[0, "source_sample"] == "G-1"
