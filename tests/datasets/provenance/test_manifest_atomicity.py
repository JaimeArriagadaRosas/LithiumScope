"""Persistence failure must preserve the previous manifest."""
from pathlib import Path
import pandas as pd
import pytest

from lithiumscope.datasets.provenance.manifest import build_tabular_manifest, write_manifest


def test_failed_replace_preserves_destination_and_cleans_temp(tmp_path, monkeypatch):
    src = tmp_path / "source.csv"
    pd.DataFrame({"Li_icpms": [1.], "SiO2": [60.]}).to_csv(src, index=False)
    manifest = build_tabular_manifest(src, "test", "model_1", "test")
    destination = tmp_path / "manifest.json"
    destination.write_bytes(b"previous")
    def fail(self, target):
        raise OSError("injected failure")
    monkeypatch.setattr(Path, "replace", fail)
    with pytest.raises(OSError, match="injected"):
        write_manifest(manifest, destination)
    assert destination.read_bytes() == b"previous"
    assert not list(tmp_path.glob(".manifest.json.*.tmp"))
