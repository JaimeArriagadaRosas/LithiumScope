"""Offline acquisition and demonstration checks with controlled provider responses."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lithiumscope.prediction.demonstration import automatic, cases
from lithiumscope.prediction.execution import session
from lithiumscope.prediction.inputs import imagery
from lithiumscope.core.hashing import file_sha256


def demo_rows():
    return pd.DataFrame({
        "case_id": ["a", "b", "c"], "Li_icpms": [10., 20., 30.],
        "Longitude": [-68., -69., -70.], "Latitude": [-23., -24., -25.],
    })


def test_demo_paths_and_required_columns(monkeypatch, tmp_path):
    config = {"demonstration": {"cases_path": "cases.csv", "imagery_cache_dir": "cache"}}
    monkeypatch.setattr(cases, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(cases, "load_config", lambda _: config)
    demo_rows().to_csv(tmp_path / "cases.csv", index=False)
    assert cases.demonstration_cases_path() == tmp_path / "cases.csv"
    assert cases.demonstration_imagery_cache() == tmp_path / "cache"
    pd.testing.assert_frame_equal(cases.load_demonstration_cases(), demo_rows())
    demo_rows().drop(columns="Latitude").to_csv(tmp_path / "cases.csv", index=False)
    with pytest.raises(RuntimeError, match="Latitude"):
        cases.load_demonstration_cases()
    config["demonstration"]["cases_path"] = str(tmp_path / "absolute.csv")
    assert cases.demonstration_cases_path() == tmp_path / "absolute.csv"


def test_demo_overlap_missing_sources_and_valid_audit(monkeypatch, tmp_path):
    monkeypatch.setattr(cases, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(cases, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(cases, "load_config", lambda _: {
        "training": {"expansion": {"output_manifest_path": "expanded.csv"}},
    })
    external = demo_rows()
    result = cases.audit_demo_overlap(external)
    assert result["sample_id_matches"] is None
    assert result["model_2_independence"]["external_independence_verified"] is False
    training = tmp_path / "data/raw/model_1/Mamani09_Table_DR2.csv"
    training.parent.mkdir(parents=True)
    external.rename(columns={"case_id": "Sample"}).to_csv(training, index=False)
    external.to_csv(tmp_path / "expanded.csv", index=False)
    result = cases.audit_demo_overlap(external)
    assert result["sample_id_matches"] == 3
    assert result["coordinate_matches"] == 3
    assert result["model_2_independence"]["possible_duplicate_count"] == 3
    assert result["model_2_independence"]["external_independence_verified"] is False
    distant = external.assign(Longitude=-50., Latitude=-10., Li_icpms=100.)
    distant.to_csv(tmp_path / "expanded.csv", index=False)
    result = cases.audit_demo_overlap(external)
    assert result["model_2_independence"]["possible_duplicate_count"] == 0
    assert result["model_2_independence"]["external_independence_verified"] is False
    distant.drop(columns="Latitude").to_csv(tmp_path / "expanded.csv", index=False)
    result = cases.audit_demo_overlap(external)
    assert "error" in result["model_2_independence"]
    assert result["model_2_independence"]["external_independence_verified"] is False


class FakeSpinner:
    def __init__(self, message):
        self.messages = [message]

    def start(self):
        return self

    def update(self, message):
        self.messages.append(message)

    def succeed(self, message):
        self.messages.append(message)

    def fail(self, message):
        self.messages.append(message)


def offline_provider(monkeypatch):
    events = []
    class Provider:
        def __init__(self, config):
            events.append(config.datetime)

        def validate(self):
            events.append("validated")

    monkeypatch.setattr(imagery, "Sentinel2Provider", Provider)
    monkeypatch.setattr(imagery, "Spinner", FakeSpinner)
    return events


def test_acquisition_keeps_success_and_per_case_failures(monkeypatch, tmp_path):
    events = offline_provider(monkeypatch)
    def download(**kwargs):
        name = kwargs["sample_id"]
        if name == "b":
            raise imagery.SentinelSceneUnavailable("no scene")
        if name == "c":
            raise imagery.SentinelAssetError("bad asset")
        destination = kwargs["destination"]
        np.save(destination, np.zeros((6, 4, 4)))
        return destination, {"scene_id": "scene-a", "cloud_cover": 2., "datetime": "2024-01-01"}

    monkeypatch.setattr(imagery, "download_patch", download)
    rows = demo_rows().rename(columns={"Longitude": "lng", "Latitude": "lat"})
    result = imagery.prepare_case_images(rows, tmp_path, datetime_override="2024-01-01/2024-12-31")
    assert events == ["2024-01-01/2024-12-31", "validated"]
    assert result["sentinel_status"].tolist() == ["ready", "no_scene", "asset_error"]
    assert Path(result.iloc[0]["image_path"]).is_file()
    assert result.iloc[0]["sentinel_scene_id"] == "scene-a"
    assert result.iloc[1]["sentinel_error"] == "no scene"
    assert result.iloc[2]["sentinel_error"] == "bad asset"


def test_acquisition_rejects_missing_coordinates_and_stops_on_provider_error(monkeypatch, tmp_path):
    offline_provider(monkeypatch)
    with pytest.raises(ValueError, match="latitud"):
        imagery.prepare_case_images(demo_rows().drop(columns="Latitude"), tmp_path)
    seen = []
    def fail(**kwargs):
        seen.append(kwargs["sample_id"])
        raise imagery.SentinelProviderError("offline")

    monkeypatch.setattr(imagery, "download_patch", fail)
    with pytest.raises(imagery.SentinelProviderError, match="offline"):
        imagery.prepare_case_images(demo_rows(), tmp_path)
    assert seen == ["a"]


def test_automatic_demo_passes_provenance_and_audit(monkeypatch, tmp_path):
    monkeypatch.setattr(session, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(automatic, "PROJECT_ROOT", tmp_path)
    data = demo_rows()
    source = tmp_path / "cases.csv"
    data.to_csv(source, index=False)
    config = {"cases_path": "cases.csv", "source_name": "fixture",
              "source_repository": "fixture", "source_commit": "abc",
              "source_path": "cases.csv", "source_license": "test",
              "sentinel_datetime": "2024-01-01/2024-12-31"}
    monkeypatch.setattr(automatic, "load_config", lambda _: {"demonstration": config})
    monkeypatch.setattr(automatic, "load_demonstration_cases", lambda: data.copy())
    monkeypatch.setattr(automatic, "demonstration_imagery_cache", lambda: tmp_path / "cache")
    audit = {"status": "unverified"}
    monkeypatch.setattr(automatic, "audit_demo_overlap", lambda frame: audit)
    def images(frame, cache, *, datetime_override):
        assert cache == tmp_path / "cache"
        assert datetime_override == config["sentinel_datetime"]
        return frame.assign(sentinel_status="no_scene")
    monkeypatch.setattr(automatic, "prepare_case_images", images)
    monkeypatch.setattr(automatic, "_run_integrated", lambda **kw: kw)
    result = automatic.run_automatic_demonstration()
    assert result["session"].mode == "demonstration"
    assert result["overlap_audit"] == audit
    assert result["input_info"]["demonstration"]["cases_sha256"] == file_sha256(source)
    pd.testing.assert_frame_equal(pd.read_csv(result["session"].inputs / "demonstration_cases.csv"), data)
