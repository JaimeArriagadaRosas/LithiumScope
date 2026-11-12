"""Candidate-only ten-case preview: no promotion, no automatic PDF."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import joblib
import pandas as pd
import pytest

from lithiumscope.cli import predict_command
from lithiumscope.prediction import integrated, session


def test_pending_candidate_load_does_not_activate_model(monkeypatch, tmp_path):
    root = tmp_path / "models"
    run_id = "training_20261112_123451_m0300"
    directory = root / "model_2" / "trained" / run_id
    directory.mkdir(parents=True)
    path = directory / "model.joblib"
    joblib.dump({"algorithm": "random_forest", "features": []}, path)
    (directory / "metadata.json").write_text(
        json.dumps({"run_id": run_id, "algorithm": "random_forest"}),
        encoding="utf-8",
    )
    status = directory / "candidate_status.json"
    status.write_text('{"status": "pending"}', encoding="utf-8")
    sentinel = root / "active_models.json"
    sentinel.write_text('{"models": {"model_2": {"model_path": "release"}}}')
    active_before = sentinel.read_bytes()
    status_before = status.read_bytes()
    monkeypatch.setattr(session, "MODELS_DIR", root)
    monkeypatch.setattr(
        session, "load_latest_model",
        lambda *args: pytest.fail("Candidate preview must not load active release"),
    )
    bundle, chosen, identity = session.load_model_with_identity(
        "model_2", candidate_run_id=run_id
    )
    assert chosen == path
    assert bundle["algorithm"] == "random_forest"
    assert identity["selection"] == "experimental_demonstration"
    assert identity["candidate_status"] == "pending"
    assert identity["model_sha256"]
    assert sentinel.read_bytes() == active_before
    assert status.read_bytes() == status_before

    status.write_text('{"status": "discarded"}', encoding="utf-8")
    with pytest.raises(ValueError, match="no está pendiente"):
        session.load_model_with_identity("model_2", candidate_run_id=run_id)
    with pytest.raises(ValueError):
        session.load_model_with_identity("model_2", candidate_run_id="../escape")


def test_default_model_selection_remains_published_release(monkeypatch, tmp_path):
    file = tmp_path / "model.joblib"
    file.write_bytes(b"dummy")
    monkeypatch.setattr(
        session, "load_latest_model",
        lambda group: ({"algorithm": "old_svm"}, file),
    )
    bundle, path, identity = session.load_model_with_identity("model_1")
    assert path == file
    assert bundle["algorithm"] == "old_svm"
    assert identity["selection"] == "active_release"
    assert identity["candidate_status"] is None


@pytest.mark.parametrize(
    "responses,expected_models,expect_pdf",
    [
        (["1", "2"], True, False),
        (["2", "1"], False, True),
    ],
)
def test_menu_selects_candidate_pair_or_active_release(
    monkeypatch, responses, expected_models, expect_pdf
):
    choices = iter(responses)
    monkeypatch.setattr(predict_command, "choose", lambda *args: next(choices))
    calls = []
    monkeypatch.setattr(
        integrated, "run_automatic_demonstration",
        lambda **kwargs: (calls.append(kwargs) or "test-result"),
    )
    monkeypatch.setattr(
        predict_command,
        "_print_integrated_result",
        lambda result: calls.append({"display": result}),
    )
    predict_command._run_demonstration()
    assert len(calls) == 2
    assert calls[0]["render_pdf"] is expect_pdf
    if expected_models:
        assert calls[0]["model_1_candidate_run_id"] == "training_20261106_111653_m0300"
        assert calls[0]["model_2_candidate_run_id"] == "training_20261112_123451_m0300"
    else:
        assert calls[0]["model_1_candidate_run_id"] is None
        assert calls[0]["model_2_candidate_run_id"] is None


def test_candidate_spectral_quality_abstention_is_respected(monkeypatch, tmp_path):
    one = tmp_path / "one.npy"
    two = tmp_path / "two.npy"
    cases = pd.DataFrame([
        {"case_id": "a", "sentinel_status": "ready", "image_path": str(one)},
        {"case_id": "b", "sentinel_status": "ready", "image_path": str(two)},
    ])
    def fake_score(path, *, bundle):
        if path == one:
            return (
                {
                    "prospectivity_score": None,
                    "priority": "abstencion",
                    "quality_status": "ABSTAIN_LOW_SPECTRAL_COVERAGE",
                    "joint_valid_fraction": .64,
                    "warning": "Insufficient coverage",
                },
                {"required_joint_valid_fraction": .95},
            )
        return (
            {
                "prospectivity_score": .72,
                "priority": "alta",
                "applicability_warning": "OK",
                "quality_status": "EVALUATED",
                "operating_threshold": .44,
                "algorithm": "random_forest",
            },
            {"missing_features": []},
        )
    monkeypatch.setattr(integrated, "score_model_2_image", fake_score)
    monkeypatch.setattr(integrated, "model_2_case_text", lambda row: "Exploratorio")
    scored, diag = integrated._score_model_2_cases(
        cases, {"features": ["red"], "operating_threshold": .44,
                "quality_min_joint_valid": .95}
    )
    assert scored.case_id.tolist() == ["b"]
    assert scored.prospectivity_score.tolist() == [.72]
    assert diag["requested_cases"] == 2
    assert diag["quality_abstention_count"] == 1
    assert diag["quality_abstentions"][0]["case_id"] == "a"
    assert diag["failed_cases"] == 1


def test_no_pdf_mode_keeps_all_evidence_and_manifest(monkeypatch, tmp_path):
    root = tmp_path / "demonstration_test"
    paths = {
        "root": root,
        "model_1": root / "model_1",
        "model_2": root / "model_2",
        "figures": root / "cross_model" / "figures",
        "cross_model": root / "cross_model",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    run = SimpleNamespace(run_id="demonstration_test", mode="demonstration", **paths)
    monkeypatch.setattr(integrated, "save_spatial_maps", lambda *args: [])
    monkeypatch.setattr(integrated, "save_satellite_input_preview", lambda *args: None)
    monkeypatch.setattr(integrated, "save_integrated_figures", lambda *args: [])
    monkeypatch.setattr(integrated, "write_workbook", lambda target, **kwargs: target)
    monkeypatch.setattr(integrated, "runtime_fingerprint", lambda: {})
    def prevent_pdf(*args, **kwargs):
        pytest.fail("PDF renderer must not run in preview mode")
    monkeypatch.setattr(integrated, "write_pdf_report", prevent_pdf)
    frame = pd.DataFrame([{"case_id": "demo_1", "Li_icpms": 11.0}])
    report, workbook, manifest = integrated._write_outputs(
        session=run,
        model_1_predictions=frame,
        model_2_predictions=frame,
        model_2_cases=pd.DataFrame([{"case_id": "demo_1", "sentinel_status": "ready"}]),
        paired=frame,
        correlations=pd.DataFrame(),
        concordance=pd.DataFrame(),
        training_vs_external=pd.DataFrame(),
        model_1_metrics={},
        model_2_metrics={},
        model_1_identity={"run_id": "new_m1"},
        model_2_identity={"run_id": "new_m2"},
        model_1_diagnostics={},
        model_2_diagnostics={"quality_abstention_count": 0},
        input_info={},
        overlap_audit=None,
        interpretation="Exploratoria",
        run_log_path=root / "prediction.log",
        lithium_threshold=21.55,
        probability_threshold=.4415,
        model_2_predicted_positive_count=0,
        render_pdf=False,
    )
    assert report is None
    assert workbook == root / "evaluation.xlsx"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["pdf_rendered"] is False
    assert payload["artifacts"]["report_pdf"] is None
    assert payload["models"]["model_1"]["run_id"] == "new_m1"
    assert payload["models"]["model_2"]["run_id"] == "new_m2"
    case_audit = pd.read_csv(run.model_2 / "case_imagery_status.csv")
    assert case_audit.model_2_scoring_status.tolist() == ["SCORED"]
    assert not (root / "report.pdf").exists()


def test_no_pdf_result_does_not_open_browser(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(
        predict_command, "open_report_in_default_browser",
        lambda path: pytest.fail("Must not open PDF when none exists"),
    )
    result = SimpleNamespace(
        interpretation="Vista previa",
        session=SimpleNamespace(
            root=tmp_path,
            model_2=tmp_path / "model_2",
            run_id="demonstration_test",
            mode="demonstration",
        ),
        workbook_path=tmp_path / "evaluation.xlsx",
        manifest_path=tmp_path / "prediction_manifest.json",
        report_path=None,
        model_1_predictions=pd.DataFrame([{"case_id": "a", "Li_icpms_predicted": 12}]),
        model_2_predictions=pd.DataFrame([{"case_id": "a", "prospectivity_score": .63}]),
    )
    predict_command._print_integrated_result(result)
    output = capsys.readouterr().out
    assert "rebuild_pdf" in output
    assert "prospectivity_score" in output
    assert "Li_icpms_predicted" in output
