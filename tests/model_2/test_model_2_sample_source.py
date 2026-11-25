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


def test_source_conflicts_censoring_and_explicit_replicates(tmp_path):
    path = tmp_path / "samples.csv"
    pd.DataFrame({
        "Sample": ["A", "A", "B", "B", "C", "C", "D 1", "D_1", "E", "F", "G", "H"],
        "source_dataset": ["one"] * 7 + ["two"] + ["one"] * 4,
        "replicate_id": ["", "", "", "", "r1", "r2", "", "", "", "", "", ""],
        "Li_icpms": [99, 5, 99, 99, 3, 7, 10, 20, "<2", -1, 0, ">10"],
        "longitude": [-70] * 12, "latitude": [-20] * 12,
    }).to_csv(path, index=False)
    result = load_georeferenced_li_samples(path)
    assert sorted(result.Li_icpms.tolist()) == [0, 3, 7, 10, 20, 99]
    assert result.sample_id.is_unique
    audit = pd.DataFrame(result.attrs["source_reconciliation"])
    assert audit.reason.eq("conflicting_measurements").sum() == 2
    assert audit.reason.eq("identical_duplicate").sum() == 1
    assert audit.censor_type.isin(["above", "below"]).sum() == 2
    assert audit.included.sum() == len(result)


def test_source_scoped_id_does_not_depend_on_other_sources(tmp_path):
    path = tmp_path / "sources.csv"
    frame = pd.DataFrame({"source_dataset": ["one", "two"], "Sample": ["A", "A"],
                          "Li_icpms": [5., 7.], "longitude": [-70., -70.],
                          "latitude": [-20., -20.], "source_li_unit": ["ppm", "percent"]})
    frame.to_csv(path, index=False)
    first = load_georeferenced_li_samples(path)
    assert len(first) == 1
    assert first.attrs["source_reconciliation"][1]["unit_status"] == "incompatible"
    frame.iloc[:1].to_csv(path, index=False)
    assert load_georeferenced_li_samples(path).sample_id.tolist() == first.sample_id.tolist()
