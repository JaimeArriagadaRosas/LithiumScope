import pytest

from lithiumscope.datasets import local_sources


def test_m1_diagnostic_uses_existing_combined_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(local_sources, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(local_sources, "load_config", lambda _: {
        "data_sources": {"georoc": {"enabled": True, "combined_path": "combined.csv"}}
    })
    with pytest.raises(FileNotFoundError):
        local_sources.local_training_source("model_1")
    source = tmp_path / "combined.csv"
    source.write_text("Li_icpms\\n10")
    original = source.read_bytes()
    assert local_sources.local_training_source("model_1") == source
    assert source.read_bytes() == original


def test_expanded_m2_diagnostic_never_falls_back_to_other_cohort(tmp_path, monkeypatch):
    monkeypatch.setattr(local_sources, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(local_sources, "load_config", lambda _: {
        "training": {"expansion": {"enabled": True, "output_manifest_path": "expanded.csv"}}
    })
    with pytest.raises(FileNotFoundError):
        local_sources.local_training_source("model_2")


def test_diagnostic_rejects_source_in_other_workspace(tmp_path, monkeypatch):
    current = tmp_path / "new"
    current.mkdir()
    old = tmp_path / "old.csv"
    old.write_text("Li")
    monkeypatch.setattr(local_sources, "PROJECT_ROOT", current)
    with pytest.raises(ValueError, match="workspace"):
        local_sources._existing_local(old)
