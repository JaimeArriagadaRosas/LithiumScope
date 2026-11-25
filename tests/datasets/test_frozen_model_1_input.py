import hashlib

import pytest

from lithiumscope.datasets.management import model_1_sources


def test_verified_frozen_source_is_reused_without_reconstruction(tmp_path, monkeypatch):
    source = tmp_path / "combined.csv"
    original = b"Li_icpms\n10\n"
    source.write_bytes(original)
    config = {"data_sources": {"georoc": {"enabled": True,
        "combined_path": str(source), "frozen_combined_sha256": hashlib.sha256(original).hexdigest()}}}
    monkeypatch.setattr(model_1_sources, "load_config", lambda _: config)
    assert model_1_sources.prepare_model_1_training_source(tmp_path / "absent.csv") == source
    assert source.read_bytes() == original


@pytest.mark.parametrize("missing", [True, False])
def test_frozen_missing_or_changed_source_fails_before_acquisition(tmp_path, monkeypatch, missing):
    source = tmp_path / "combined.csv"
    if not missing:
        source.write_bytes(b"changed")
    config = {"data_sources": {"georoc": {"enabled": True,
        "combined_path": str(source), "frozen_combined_sha256": "a" * 64}}}
    monkeypatch.setattr(model_1_sources, "load_config", lambda _: config)
    with pytest.raises(FileNotFoundError if missing else ValueError, match="Frozen"):
        model_1_sources.prepare_model_1_training_source(tmp_path / "absent.csv")
    assert source.exists() is (not missing)
