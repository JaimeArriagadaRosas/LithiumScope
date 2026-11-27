import json
from pathlib import Path

import pandas as pd
import pytest

from lithiumscope.core.hashing import file_sha256
from lithiumscope.datasets.adapters.georoc.ingestion import ingest_georoc_files
from lithiumscope.datasets.provenance.publication import publish_pair


def test_ingestion_records_encoding_and_source_identity(tmp_path):
    source = tmp_path / "raw.csv"
    source.write_bytes("MATERIAL,LI(PPM),LONGITUDE,LATITUDE,SAMPLE NAME\nWR,10,-70,-30,Andés\n".encode("cp1252"))
    result = ingest_georoc_files([source], tmp_path / "out.csv", minimum_predictors=0)
    audit = json.loads(result.audit_path.read_text(encoding="utf-8"))
    assert audit["output_sha256"] == file_sha256(result.path)
    assert audit["source_details"][0]["encoding"] == "cp1252"
    assert audit["complete_source_coverage"]
    assert pd.read_csv(result.path).source_sample.iloc[0] == "Andés"


def test_incompatible_source_never_silently_disappears(tmp_path):
    source = tmp_path / "good.csv"
    source.write_text("MATERIAL,LI(PPM),LONGITUDE,LATITUDE\nWR,10,-70,-30\n")
    bad = tmp_path / "bad.csv"
    bad.write_text("MATERIAL,NOTE\nWR,test\n")
    dest = tmp_path / "out.csv"
    with pytest.raises(ValueError, match="required columns"):
        ingest_georoc_files([source, bad], dest, minimum_predictors=0)
    assert not dest.exists()
    result = ingest_georoc_files([source, bad], dest, minimum_predictors=0, allow_incompatible_sources=True)
    audit = json.loads(result.audit_path.read_text(encoding="utf-8"))
    assert not audit["complete_source_coverage"]
    assert audit["source_details"][1]["state"] == "skipped_schema"


def test_pair_rolls_back_on_failed_evidence_replace(tmp_path, monkeypatch):
    data, evidence, new_data, new_evidence = [tmp_path / n for n in ("data", "evidence", "new_data", "new_evidence")]
    for path, value in [(data, "old"), (evidence, "old manifest"), (new_data, "new"), (new_evidence, "new manifest")]:
        path.write_text(value)
    replace = Path.replace
    def fail(self, target):
        if self == new_evidence:
            raise OSError("injected")
        return replace(self, target)
    monkeypatch.setattr(Path, "replace", fail)
    with pytest.raises(OSError, match="injected"):
        publish_pair(new_data, data, new_evidence, evidence)
    assert data.read_text() == "old"
    assert evidence.read_text() == "old manifest"
