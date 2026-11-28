from pathlib import Path

import pandas as pd

from lithiumscope.datasets.adapters.georoc.ingestion import (
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


def test_georoc_ingestion_supports_material_and_coordinate_ranges(
    tmp_path: Path,
):
    source = tmp_path / "georoc_ranges.csv"
    pd.DataFrame(
        [
            {
                "SAMPLE NAME": "G-2",
                "MATERIAL": "WHOLE ROCK",
                "LONGITUDE MIN": -70.2,
                "LONGITUDE MAX": -70.0,
                "LATITUDE MIN": -23.6,
                "LATITUDE MAX": -23.4,
                "LI": 20.0,
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

    destination = tmp_path / "harmonized_ranges.csv"
    result = ingest_georoc_files(
        [source],
        destination,
        chunksize=1,
        minimum_predictors=8,
    )

    frame = pd.read_csv(result.path)
    assert result.rows_kept == 1
    assert frame.loc[0, "Sample_type"] == "WHOLE ROCK"
    assert frame.loc[0, "Longitude"] == -70.1
    assert frame.loc[0, "Latitude"] == -23.5
    assert frame.loc[0, "Li_icpms"] == 20.0



def test_georoc_ingestion_parses_batch_material_and_preserves_traceability(
    tmp_path: Path,
):
    source = tmp_path / "georoc_batch_material.csv"
    pd.DataFrame(
        [
            {
                "SAMPLE NAME": "WR-1",
                "MATERIAL": "/ WR [13695]",
                "LONGITUDE": -70.1,
                "LATITUDE": -23.4,
                "LI": 15.0,
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
            },
            {
                "SAMPLE NAME": "GL-1",
                "MATERIAL": "/ GL [10206]",
                "LONGITUDE": -69.9,
                "LATITUDE": -22.0,
                "LI": 25.0,
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
            },
        ]
    ).to_csv(source, index=False)

    destination = tmp_path / "harmonized_batch.csv"
    result = ingest_georoc_files(
        [source],
        destination,
        chunksize=10,
        minimum_predictors=8,
        allowed_material_types=("WHOLE ROCK",),
    )

    frame = pd.read_csv(result.path)

    assert result.rows_read == 2
    assert result.rows_kept == 1
    assert frame.loc[0, "source_sample"] == "WR-1"
    assert frame.loc[0, "Sample_type"] == "WHOLE ROCK"
    assert frame.loc[0, "Sample_type_raw"] == "/ WR [13695]"
    assert frame.loc[0, "Sample_material_code"] == "WR"
    assert str(frame.loc[0, "Sample_material_batch_id"]) == "13695"


def test_research_view_preserves_original_and_separates_footer(tmp_path):
    from lithiumscope.datasets.adapters.georoc.research_view import normalize_research_export
    source = tmp_path / "raw.csv"
    source.write_text('SAMPLE NAME,LI(PPM),,SIO2(WT%)\nA,15,,60\nReferences:\n"[42] citation",extra\n')
    before = source.read_bytes()
    destination = tmp_path / "view.csv"
    report = normalize_research_export(source, destination)
    assert source.read_bytes() == before
    assert report["empty_auxiliary_columns_1based"] == [3]
    assert report["data_records"] == 1
    assert len(report["footer"]) == 2
    assert list(pd.read_csv(destination).columns) == ["SAMPLE NAME", "LI(PPM)", "SIO2(WT%)"]


def test_research_view_rejects_populated_blank_and_malformed_rows(tmp_path):
    import pytest
    from lithiumscope.core.exceptions import InputValidationError
    from lithiumscope.datasets.adapters.georoc.research_view import normalize_research_export
    source = tmp_path / "raw.csv"
    for i, row in enumerate(["A,15,unmapped", "A,15"]):
        source.write_text("SAMPLE NAME,LI(PPM),\n" + row + "\n")
        with pytest.raises(InputValidationError):
            normalize_research_export(source, tmp_path / f"view_{i}.csv")


def test_research_view_recognizes_multicell_bibliography_marker(tmp_path):
    from lithiumscope.datasets.adapters.georoc.research_view import normalize_research_export
    source = tmp_path / "raw.csv"
    source.write_text('SAMPLE NAME,LI(PPM),,SIO2(WT%)\nA,15,,60\nReferences:,"[42] Author (2020)",journal\n"[43] Author (2021)"\n')
    report = normalize_research_export(source, tmp_path / "view.csv")
    assert report["data_records"] == 1
    assert len(report["footer"]) == 2
    assert report["footer"][0]["cells"][1] == "[42] Author (2020)"


def test_research_view_blocks_scientific_data_after_footer_and_reports_width(tmp_path):
    import pytest
    from lithiumscope.core.exceptions import InputValidationError
    from lithiumscope.datasets.adapters.georoc.research_view import normalize_research_export
    source = tmp_path / "raw.csv"
    header = "SAMPLE NAME,LI(PPM),,SIO2(WT%)\nA,15,,60\n"
    source.write_text(header + "References:\nB,20,,65\n")
    with pytest.raises(InputValidationError, match="Scientific-looking row after bibliography"):
        normalize_research_export(source, tmp_path / "footer.csv")
    source.write_text(header + "B,20\n")
    with pytest.raises(InputValidationError, match=r"raw.csv: Malformed GEOROC row at line 3; expected=4, actual=2"):
        normalize_research_export(source, tmp_path / "width.csv")
