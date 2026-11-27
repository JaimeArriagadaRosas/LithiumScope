"""Regression contracts for the extracted GEOROC stages."""
from unittest.mock import Mock
import json

import pandas as pd
import pytest

from lithiumscope.datasets.adapters.georoc import ingestion
from lithiumscope.datasets.adapters.georoc.transformation import harmonize_chunk


def source_frame():
    return pd.DataFrame({
        "MATERIAL": ["WHOLE ROCK", "VOLCANIC GLASS", "WR", "WR"],
        "LI(PPM)": ["<10", "12", "15", "-2"],
        "LONGITUDE": [-70, -70, -71, -72],
        "LATITUDE": [-20, -20, -21, -22],
        "SIO2(WT%)": [60, 61, 62, 63],
        "FEOT(WT%)": [4, 5, 6, 7],
    })


def test_chunk_size_does_not_change_scientific_output(tmp_path, monkeypatch):
    source = tmp_path / "input.csv"
    source_frame().to_csv(source, index=False)
    log = Mock()
    monkeypatch.setattr(ingestion, "logger", log)
    a = ingestion.ingest_georoc_files([source], tmp_path / "a.csv", chunksize=1,
                                      minimum_predictors=1, iron_total_mode="feot_equivalent")
    b = ingestion.ingest_georoc_files([source], tmp_path / "b.csv", chunksize=20,
                                      minimum_predictors=1, iron_total_mode="feot_equivalent")
    pd.testing.assert_frame_equal(pd.read_csv(a.path), pd.read_csv(b.path))
    rows = pd.read_csv(a.path)
    assert rows.Li_icpms.tolist() == [5., 15.]
    assert rows.Fe2O3.tolist() == pytest.approx([4*1.111348, 6*1.111348])
    assert json.loads(a.audit_path.read_text())["rows_kept"] == 2
    assert log.info.call_count >= 4


def test_missing_material_rejects_all_without_fabricating_identity():
    chunk = pd.DataFrame({"LI": ["10"]})
    frame, rejected = harmonize_chunk(chunk, column_map={"LI":"Li_icpms"},
        source_name="test", source_file="source.csv", allowed_material_types=("WR",),
        minimum_predictors=0)
    assert frame.empty and rejected == 1
    assert "source_sample" not in frame


def test_ingestion_failure_removes_partial_output(tmp_path, monkeypatch):
    source = tmp_path / "input.csv"
    source_frame().to_csv(source, index=False)
    dest = tmp_path / "combined.csv"
    dest.write_bytes(b"previous")
    original = ingestion.harmonize_chunk
    calls = 0
    def fail_on_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected chunk failure")
        return original(*args, **kwargs)
    log = Mock()
    monkeypatch.setattr(ingestion, "harmonize_chunk", fail_on_second)
    monkeypatch.setattr(ingestion, "logger", log)
    with pytest.raises(RuntimeError, match="injected"):
        ingestion.ingest_georoc_files([source], dest, chunksize=1, minimum_predictors=1)
    assert dest.read_bytes() == b"previous"
    assert not dest.with_suffix(".csv.tmp").exists()
    log.exception.assert_called_once()
