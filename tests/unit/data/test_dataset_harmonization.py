import pandas as pd

from lithiumscope.datasets.adapters.georoc import (
    GeorocAdapterConfig,
    harmonize_georoc_frame,
)
from lithiumscope.datasets.harmonization import (
    merge_harmonized_sources,
)


def test_georoc_adapter_requires_explicit_mapping():
    raw = pd.DataFrame(
        [
            {
                "LI": 12.5,
                "LONG": -70.1,
                "LAT": -23.4,
                "SIO2": 65.0,
            }
        ]
    )
    config = GeorocAdapterConfig(
        column_map={
            "LI": "Li_icpms",
            "LONG": "Longitude",
            "LAT": "Latitude",
            "SIO2": "SiO2",
        },
        source_name="GEOROC test",
    )

    result = harmonize_georoc_frame(
        raw,
        config,
    )

    assert result.loc[0, "Li_icpms"] == 12.5
    assert result.loc[0, "Longitude"] == -70.1
    assert result.loc[0, "source_dataset"] == "GEOROC test"


def test_merge_harmonized_sources_records_deduplication():
    first = pd.DataFrame(
        [
            {
                "source_sample": "A",
                "Longitude": -70.0,
                "Latitude": -20.0,
                "Li_icpms": 10.0,
            }
        ]
    )
    duplicate = first.copy()

    merged, audit = merge_harmonized_sources(
        [first, duplicate]
    )

    assert len(merged) == 1
    assert audit["duplicates_removed"] == 1


def test_merge_coalesces_identifiers_across_sources():
    mamani = pd.DataFrame(
        [
            {
                "Sample": "A-1",
                "Longitude": -70.0,
                "Latitude": -20.0,
                "Li_icpms": 10.0,
            }
        ]
    )
    georoc = pd.DataFrame(
        [
            {
                "source_sample": "A-1",
                "Longitude": -70.0,
                "Latitude": -20.0,
                "Li_icpms": 10.0,
            }
        ]
    )

    merged, audit = merge_harmonized_sources(
        [mamani, georoc]
    )

    assert len(merged) == 1
    assert audit["duplicates_removed"] == 1
