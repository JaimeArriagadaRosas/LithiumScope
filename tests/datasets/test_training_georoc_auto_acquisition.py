from types import SimpleNamespace

import pandas as pd
import pytest

from lithiumscope.datasets import georoc_filtered_acquisition as georoc
from lithiumscope.datasets import model_1_sources as sources


def setup(monkeypatch, tmp_path):
    monkeypatch.setattr(sources, "_project_path", lambda value: tmp_path / value)
    monkeypatch.setattr(
        sources, "load_config",
        lambda name: {"data_sources": {"georoc": {
            "enabled": True,
            "input_glob": "raw/georoc/*.csv",
            "harmonized_path": "interim/georoc_harmonized.csv",
            "combined_path": "processed/combined.csv",
            "audit_path": "processed/merge_audit.json",
            "minimum_predictors": 8,
        }}},
    )
    base = tmp_path / "mamani.csv"
    base.write_text("Li_icpms" + chr(10) + "10" + chr(10), encoding="utf-8")
    return base, tmp_path / "raw/georoc"


def stub_merger(monkeypatch, tmp_path):
    def ingest(files, target, **kwargs):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("Li_icpms" + chr(10) + "11" + chr(10), encoding="utf-8")
        return SimpleNamespace(
            path=target, files=tuple(str(x) for x in files),
            rows_read=1, rows_kept=1, audit_path=tmp_path/"audit.json",
        )
    monkeypatch.setattr(sources, "ingest_georoc_files", ingest)
    monkeypatch.setattr(
        sources, "load_data",
        lambda path, quiet=True: pd.DataFrame({"Li_icpms": [10.0]}),
    )
    monkeypatch.setattr(
        sources, "merge_harmonized_sources",
        lambda frames: (
            pd.DataFrame({"Li_icpms": [10.0, 11.0]}),
            {"duplicates_removed": 0},
        ),
    )


def test_missing_georoc_calls_original_filtered_download_and_merges(monkeypatch, tmp_path):
    base, folder = setup(monkeypatch, tmp_path)
    stub_merger(monkeypatch, tmp_path)
    calls = []
    def acquire(destination):
        calls.append(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("LI,LONGITUDE,LATITUDE" + chr(10) +
                               "10,-70,-23" + chr(10), encoding="utf-8")
        return destination
    monkeypatch.setattr(georoc, "acquire_filtered_georoc", acquire)
    combined = sources.prepare_model_1_training_source(base, include_georoc=True)
    assert calls == [folder / georoc.GEOROC_FILTERED_NAME]
    assert (folder / "lithiumscope_georoc_query_contract.json").is_file()
    assert len(pd.read_csv(combined)) == 2
    assert (tmp_path / "processed/merge_audit.json").is_file()


def test_existing_georoc_skips_downloader(monkeypatch, tmp_path):
    base, folder = setup(monkeypatch, tmp_path)
    stub_merger(monkeypatch, tmp_path)
    folder.mkdir(parents=True)
    (folder / "approved.csv").write_text("LI" + chr(10) + "10" + chr(10))
    monkeypatch.setattr(
        georoc, "acquire_filtered_georoc",
        lambda *args, **kwargs: pytest.fail("Existing GEOROC must be reused"),
    )
    assert sources.prepare_model_1_training_source(base, include_georoc=True).exists()


def test_failed_download_does_not_fallback_to_mamani(monkeypatch, tmp_path):
    base, folder = setup(monkeypatch, tmp_path)
    def fail(*args, **kwargs):
        raise RuntimeError("GEOROC unavailable")
    monkeypatch.setattr(georoc, "acquire_filtered_georoc", fail)
    with pytest.raises(RuntimeError, match="GEOROC unavailable"):
        sources.prepare_model_1_training_source(base, include_georoc=True)
    assert not (tmp_path / "processed/combined.csv").exists()
