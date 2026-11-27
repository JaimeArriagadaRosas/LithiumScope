"""Regression contracts for workbook, portable evidence and report recovery."""
import json
import shutil

import pandas as pd
import pytest

from lithiumscope.reporting import rebuild_pdf
from lithiumscope.reporting.workbook import write_workbook
from lithiumscope.results.prediction_artifacts import artifact_hashes, verify_report_sources


def test_workbook_contains_all_prediction_tables(tmp_path):
    frame = pd.DataFrame({"value": [1, 2]})
    destination = tmp_path / "evaluation.xlsx"
    write_workbook(destination, model_1=frame, model_2=frame, paired=frame,
                   correlations=frame, concordance=frame, training_vs_external=frame)
    with pd.ExcelFile(destination) as workbook:
        assert set(workbook.sheet_names) == {
            "modelo_1", "modelo_2", "casos_emparejados", "correlaciones",
            "concordancia", "training_vs_external",
        }
        assert workbook.parse("modelo_1")["value"].tolist() == [1, 2]


def test_workbook_failure_does_not_replace_previous_file(tmp_path, monkeypatch):
    destination = tmp_path / "evaluation.xlsx"
    destination.write_bytes(b"previous")
    def fail(*args, **kwargs):
        raise OSError("injected")
    monkeypatch.setattr(pd, "ExcelWriter", fail)
    frame = pd.DataFrame()
    with pytest.raises(OSError, match="injected"):
        write_workbook(destination, model_1=frame, model_2=frame, paired=frame,
                       correlations=frame, concordance=frame, training_vs_external=frame)
    assert destination.read_bytes() == b"previous"
    assert list(tmp_path.iterdir()) == [destination]


def fixture_session(root):
    artifacts = {
        "model_1_predictions": "model_1/predictions.csv",
        "model_2_predictions": "model_2/predictions.csv",
        "joined_cases": "cross_model/joined_cases.csv",
        "correlations": "cross_model/correlations.csv",
        "concordance": "cross_model/concordance.csv",
        "training_vs_external": "training_vs_external.csv",
        "interpretation": "interpretation.txt",
        "model_1_diagnostics": "model_1/diagnostics.json",
        "model_2_diagnostics": "model_2/diagnostics.json",
        "report_pdf": "report.pdf",
        "figures": [],
        "maps": [],
        "decision_previews": [],
    }
    for relative in artifacts.values():
        if not isinstance(relative, str):
            continue
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}" if path.suffix == ".json" else "case_id\na\n", encoding="utf-8")
    manifest = {
        "schema_version": 2, "mode": "demonstration", "title": "Experimental",
        "run_id": "test", "artifacts": artifacts, "artifact_sha256": artifact_hashes(root, artifacts),
    }
    (root / "prediction_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def test_versioned_evidence_survives_relocation_and_detects_tampering(tmp_path):
    original = tmp_path / "original"
    manifest = fixture_session(original)
    copied = tmp_path / "moved"
    shutil.copytree(original, copied)
    verify_report_sources(copied, manifest)
    (copied / "model_1/predictions.csv").write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="evidence differs"):
        verify_report_sources(copied, manifest)


def test_regeneration_restores_previous_pdf_if_manifest_commit_fails(tmp_path, monkeypatch):
    root = tmp_path / "run"
    fixture_session(root)
    before = (root / "report.pdf").read_bytes()
    manifest_before = (root / "prediction_manifest.json").read_bytes()
    def render(path, **context):
        assert context["title"] == "Experimental"
        path.write_bytes(b"%PDF-new")
        return path
    def fail(*args, **kwargs):
        raise OSError("manifest write failed")
    monkeypatch.setattr(rebuild_pdf, "write_pdf_report", render)
    monkeypatch.setattr(rebuild_pdf, "write_legacy_evidence_json", fail)
    with pytest.raises(OSError, match="manifest write failed"):
        rebuild_pdf.rebuild_pdf(root)
    assert (root / "report.pdf").read_bytes() == before
    assert (root / "prediction_manifest.json").read_bytes() == manifest_before
    assert not list(root.glob(".report.*"))
