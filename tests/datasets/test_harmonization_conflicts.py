import pandas as pd

from lithiumscope.datasets.harmonization import deduplicate_harmonized_sources


def test_conflicting_lithium_values_are_retained():
    frame = pd.DataFrame({"Sample": ["A", "A"], "Longitude": [-70., -70.],
                          "Latitude": [-30., -30.], "Li_icpms": [10., 20.]})
    merged, audit = deduplicate_harmonized_sources(frame)
    assert len(merged) == 2
    assert audit["events"][0]["state"] == "conflict_retained"


def test_incomplete_identity_is_never_deleted():
    frame = pd.DataFrame({"Sample": [None, None], "Longitude": [-70., -70.],
                          "Latitude": [-30., -30.], "Li_icpms": [10., 10.]})
    merged, audit = deduplicate_harmonized_sources(frame)
    assert len(merged) == 2
    assert audit["unresolved_rows_retained"] == [0, 1]


def test_identical_complete_measurements_have_explicit_lineage():
    frame = pd.DataFrame({"Sample": ["A", "A"], "Longitude": [-70., -70.],
                          "Latitude": [-30., -30.], "Li_icpms": [10., 10.]})
    merged, audit = deduplicate_harmonized_sources(frame)
    assert len(merged) == 1
    assert audit["events"] == [{"state": "identical_measurements", "kept_row": 0, "removed_row": 1}]
