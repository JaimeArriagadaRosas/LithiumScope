"""Exercise real analysis and report publication without trained model artifacts."""
from contextlib import contextmanager
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lithiumscope.prediction.execution import integrated, model_2_scoring, outputs, session
from lithiumscope.results.prediction_artifacts import verify_report_sources
from lithiumscope.reporting.rebuild_pdf import rebuild_pdf


def model_2_score(image, *, bundle):
    name = Path(image).stem
    if name == "bad":
        raise ValueError("invalid patch")
    if name == "abstain":
        return {"prospectivity_score": None, "warning": "no score", "quality_status": "invalid"}, {}
    score = {"a": .1, "b": .65, "c": .9}[name]
    return {"image": str(image), "prospectivity_score": score,
            "priority": "alta" if score >= .7 else "media" if score >= .4 else "baja",
            "operating_threshold": .6, "out_of_training_range_fraction": .1,
            "applicability_warning": "OK"}, {"missing_features": ["example"]}


def test_batch_reports_missing_bad_and_abstained_cases(monkeypatch):
    monkeypatch.setattr(model_2_scoring, "score_model_2_image", model_2_score)
    data = pd.DataFrame({
        "case_id": ["a", "bad", "abstain", "missing"],
        "image_path": ["a.npy", "bad.npy", "abstain.npy", None],
        "sentinel_status": ["ready", "ready", "ready", "no_scene"],
        "sentinel_error": [None, None, None, "no scene"],
    })
    bundle = {"features": ["example"], "operating_threshold": .6}
    predictions, diagnostic = model_2_scoring._score_model_2_cases(data, bundle)
    assert predictions["case_id"].tolist() == ["a"]
    assert diagnostic["ready_cases"] == 1
    assert diagnostic["failed_cases"] == 3
    assert diagnostic["requested_cases"] == 4
    assert diagnostic["missing_feature_counts"] == {"example": 1}
    assert {item["case_id"] for item in diagnostic["failed_case_details"]} == {"bad", "abstain", "missing"}
    with pytest.raises(ValueError, match="invalid patch"):
        model_2_scoring._score_model_2_cases(data, bundle, strict=True)
    predictions, diagnostic = model_2_scoring._score_model_2_cases(
        data[data.case_id != "bad"], bundle, strict=True, include_abstentions=True,
    )
    assert predictions["case_id"].tolist() == ["a", "abstain"]
    assert pd.isna(predictions.iloc[1]["prospectivity_score"])
    assert diagnostic["ready_cases"] == 1


@pytest.mark.parametrize("mode", ["complete", "demonstration"])
def test_integrated_prediction_writes_real_report_and_workbook(monkeypatch, tmp_path, mode):
    monkeypatch.setattr(session, "RESULTS_DIR", tmp_path / "results")
    run = session.PredictionSession.create(mode)
    frame = pd.DataFrame({
        "case_id": ["a", "b", "c"], "Li_icpms": [10., 25., 40.],
        "Longitude": [-68., -69., -70.], "Latitude": [-23., -24., -25.],
    })
    patch_dir = run.model_2 / "imagery"
    patch_dir.mkdir()
    paths = []
    for name in frame.case_id:
        path = patch_dir / f"{name}.npy"
        np.save(path, np.arange(3 * 8 * 8, dtype=np.float32).reshape(3, 8, 8))
        paths.append(str(path))
    images = frame.assign(image_path=paths, sentinel_status="ready")
    model_calls = []
    def load_model(group, *, candidate_run_id=None):
        model_calls.append((group, candidate_run_id))
        identity = {"run_id": "training_fixture_" + group, "algorithm": "fixture",
                    "model_sha256": "a" * 64, "training_metrics": {},
                    "selection": "experimental_demonstration" if mode == "demonstration" else "active_release"}
        if group == "model_2":
            identity["bands"] = ["red", "green", "blue"]
        return {"features": ["example"], "threshold_ppm": 20., "operating_threshold": .6}, None, identity

    def predict(data, *, bundle):
        return data.assign(Li_icpms_predicted=[11., 24., 41.],
                           Li_icpms_interval_low_q90=[8., 20., 38.],
                           Li_icpms_interval_high_q90=[14., 28., 44.],
                           out_of_training_range_fraction=[.1, .2, .3],
                           applicability_warning=["OK", "OK", "OUT_OF_DOMAIN"]), {"out_of_domain_rows": 1}

    @contextmanager
    def log(*args):
        yield run.root / "run.log"

    monkeypatch.setattr(integrated, "load_config", lambda _: {
        "interpretation": {"model_2_classification_threshold": .5},
        "demonstration": {"candidate_models": {"model_1": "training_m1", "model_2": "training_m2"}},
    })
    monkeypatch.setattr(integrated, "run_log", log)
    monkeypatch.setattr(integrated, "load_model_with_identity", load_model)
    monkeypatch.setattr(integrated, "predict_model_1_frame", predict)
    monkeypatch.setattr(model_2_scoring, "score_model_2_image", model_2_score)
    monkeypatch.setattr(outputs, "runtime_fingerprint", lambda: {"git_commit": "fixture"})
    if mode == "demonstration":
        from lithiumscope.prediction.analysis import model_selection_evidence
        monkeypatch.setattr(model_selection_evidence, "gather_selection_evidence", lambda *args: {"state": "UNAVAILABLE"})
    result = integrated._run_integrated(session=run, model_1_input=frame,
                                        model_2_cases=images, input_info={"fixture": True})
    assert len(result.paired) == 3
    assert result.model_1_metrics["mae"] == pytest.approx(1.)
    assert result.model_2_metrics["balanced_accuracy"] == pytest.approx(1.)
    assert result.report_path.read_bytes().startswith(b"%PDF")
    with pd.ExcelFile(result.workbook_path) as workbook:
        assert len(workbook.parse("casos_emparejados")) == 3
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    verify_report_sources(run.root, manifest)
    assert manifest["counts"]["paired_cases"] == 3
    assert manifest["counts"]["model_2_predicted_positive"] == 2
    assert len(manifest["artifacts"]["maps"]) == 2
    assert len(manifest["artifacts"]["figures"]) == 9
    assert manifest["artifacts"]["report_pdf"] == "report.pdf"
    assert json.loads((run.root / "publication_status.json").read_text())["state"] == "complete"
    if mode == "demonstration":
        assert model_calls == [("model_1", "training_m1"), ("model_2", "training_m2")]
        assert "experimental" in manifest["title"]
        assert (run.root / "model_selection_evidence.json").is_file()
    else:
        assert model_calls == [("model_1", None), ("model_2", None)]
    regenerated = rebuild_pdf(run.root)
    assert regenerated.read_bytes().startswith(b"%PDF")
    verified = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    verify_report_sources(run.root, verified)
    assert verified["title"] == manifest["title"]


def test_publication_failure_is_not_marked_complete(monkeypatch, tmp_path):
    from lithiumscope.prediction.execution.contracts import PredictionFrames, PredictionEvidence, PredictionThresholds
    from lithiumscope.reporting import workbook

    monkeypatch.setattr(session, "RESULTS_DIR", tmp_path)
    run = session.PredictionSession.create("complete")
    data = pd.DataFrame()
    frames = PredictionFrames(data, data, data, data, data, data, data)
    evidence = PredictionEvidence({}, {}, {}, {}, {}, {}, {}, None, "fixture")
    def fail(*args, **kwargs):
        raise OSError("workbook failure")
    monkeypatch.setattr(workbook, "write_workbook", fail)
    with pytest.raises(OSError, match="workbook failure"):
        outputs._write_outputs(run, frames, evidence, PredictionThresholds(20., .5, 0), run.root / "run.log")
    status = json.loads((run.root / "publication_status.json").read_text())
    assert status["state"] == "failed"
    assert "workbook failure" in status["reason"]
    assert not (run.root / "prediction_manifest.json").exists()
