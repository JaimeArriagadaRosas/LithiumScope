import pandas as pd

from lithiumscope.model_2.data.sample_source import load_georeferenced_li_samples


def test_georeferenced_samples_are_built_automatically(tmp_path):
    path = tmp_path / "samples.csv"
    pd.DataFrame(
        {
            "Sample": ["A 1", "B-2", "bad"],
            "Li_icpms": [12.0, 30.0, None],
            "Longitude (X)": [-70.5, -71.25, -200.0],
            "Latitude (Y)": [-20.25, -21.5, -10.0],
        }
    ).to_csv(path, index=False)

    result = load_georeferenced_li_samples(path)

    assert len(result) == 2
    assert list(result["sample_id"]) == ["A_1", "B-2"]
    assert result["spatial_group"].notna().all()
    assert result["Li_icpms"].tolist() == [12.0, 30.0]
