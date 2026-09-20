from pathlib import Path

import pandas as pd


def test_versioned_demo_fixture_has_multiple_external_cases():
    root = Path(__file__).resolve().parents[1]
    path = root / "examples" / "integrated_demo" / "cases.csv"
    frame = pd.read_csv(path)

    assert len(frame) == 10
    assert frame["case_id"].is_unique
    assert frame["Li_icpms"].notna().all()
    assert frame["Longitude"].notna().all()
    assert frame["Latitude"].notna().all()
    assert frame["Longitude (X)"].equals(frame["Longitude"])
    assert frame["Latitude (Y)"].equals(frame["Latitude"])
    assert frame["source_dataset"].str.contains("andes_paleoelevation").all()


def test_demo_fixture_never_uses_target_as_predictor_contract():
    root = Path(__file__).resolve().parents[1]
    frame = pd.read_csv(root / "examples" / "integrated_demo" / "cases.csv")

    assert "Li_icpms" in frame.columns
    assert "case_id" in frame.columns
    assert "Fe2O3" in frame.columns
    assert frame["Fe2O3"].isna().all()
