"""Original ten-case demonstration: only candidate model identities are different.

No additional menus, optional render flags or alternate PDF pipeline.
The default integrated prediction must continue to use the active release.
"""
from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace

import joblib
import pytest

from lithiumscope.cli import predict_command
from lithiumscope.prediction.execution import integrated, session
from lithiumscope.prediction.demonstration import automatic
from lithiumscope.reporting.service import write_pdf_report


def test_pending_model_load_does_not_promote_release(monkeypatch, tmp_path):
    root = tmp_path / "models"
    run_id = "training_20261112_123451_m0300"
    folder = root / "model_2" / "trained" / run_id
    folder.mkdir(parents=True)
    path = folder / "model.joblib"
    joblib.dump({"algorithm": "random_forest", "features": []}, path)
    (folder / "metadata.json").write_text(
        json.dumps({"run_id": run_id, "algorithm": "random_forest"}),
        encoding="utf-8",
    )
    status = folder / "candidate_status.json"
    status.write_text('{"status": "pending"}', encoding="utf-8")
    active = root / "active_models.json"
    active.write_text('{"models":{"model_2":{"model_path":"original"}}}')
    before_active = active.read_bytes()
    before_status = status.read_bytes()

    monkeypatch.setattr(session, "MODELS_DIR", root)
    monkeypatch.setattr(
        session, "load_latest_model",
        lambda group: pytest.fail("Candidate must not load active release"),
    )
    bundle, chosen, info = session.load_model_with_identity(
        "model_2", candidate_run_id=run_id
    )
    assert chosen == path
    assert bundle["algorithm"] == "random_forest"
    assert info["run_id"] == run_id
    assert info["candidate_status"] == "pending"
    assert info["selection"] == "experimental_demonstration"
    assert info["model_sha256"]
    assert active.read_bytes() == before_active
    assert status.read_bytes() == before_status

    status.write_text('{"status": "discarded"}', encoding="utf-8")
    with pytest.raises(ValueError, match="pendiente"):
        session.load_model_with_identity("model_2", candidate_run_id=run_id)
    with pytest.raises(ValueError, match="inválido"):
        session.load_model_with_identity("model_2", candidate_run_id="../escape")


def test_normal_load_keeps_active_release(monkeypatch, tmp_path):
    active = tmp_path / "model.joblib"
    active.write_bytes(b"placeholder")
    monkeypatch.setattr(
        session, "load_latest_model",
        lambda group: ({"algorithm": "release_model"}, active),
    )
    bundle, path, info = session.load_model_with_identity("model_1")
    assert path == active
    assert bundle["algorithm"] == "release_model"
    assert info["selection"] == "active_release"
    assert info["candidate_status"] is None


def test_original_demonstration_menu_has_no_extra_choices(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(predict_command, "choose", lambda *a, **kw: "2")
    monkeypatch.setattr(predict_command, "pick_file", lambda *a, **kw: pytest.fail("Demonstration must not request files"))
    monkeypatch.setattr(
        automatic, "run_automatic_demonstration",
        lambda: calls.append("demonstration") or object(),
    )
    monkeypatch.setattr(
        predict_command, "_print_integrated_result",
        lambda result: calls.append("PDF_output_original"),
    )

    predict_command.run()
    out = capsys.readouterr().out
    assert calls == ["demonstration", "PDF_output_original"]
    assert "Demostración integrada automática" in out
    assert "Seleccione modelos" not in out
    assert "Seleccione salida" not in out
    assert "SIN PDF" not in out


@pytest.mark.parametrize(
    "mode,expected_m1,expected_m2",
    [
        (
            "demonstration",
            "training_20261106_111653_m0300",
            "training_20261112_123451_m0300",
        ),
        ("complete", None, None),
    ],
)
def test_candidate_selection_is_scoped_to_demo_only(
    monkeypatch, tmp_path, mode, expected_m1, expected_m2
):
    class AfterModelLoads(Exception):
        pass

    calls = []

    def fake_loader(group, *, candidate_run_id=None):
        calls.append((group, candidate_run_id))
        if group == "model_2":
            raise AfterModelLoads()
        return {}, Path("unused.joblib"), {"run_id": "test"}

    monkeypatch.setattr(integrated, "load_model_with_identity", fake_loader)
    monkeypatch.setattr(
        integrated, "load_config",
        lambda name: {
            "interpretation": {"model_2_classification_threshold": 0.5},
            "demonstration": {
                "candidate_models": {
                    "model_1": "training_20261106_111653_m0300",
                    "model_2": "training_20261112_123451_m0300",
                }
            },
        },
    )

    @contextmanager
    def fake_log(*args, **kwargs):
        yield tmp_path / "prediction.log"

    monkeypatch.setattr(integrated, "run_log", fake_log)
    mock_run = SimpleNamespace(mode=mode, run_id="test", root=tmp_path)
    with pytest.raises(AfterModelLoads):
        integrated._run_integrated(
            session=mock_run,
            model_1_input=None,
            model_2_cases=None,
            input_info={},
        )
    assert calls == [
        ("model_1", expected_m1),
        ("model_2", expected_m2),
    ]


def test_pdf_is_required_and_legacy_pdf_renderer_remains_used():
    from inspect import signature

    assert "render_pdf" not in signature(integrated._run_integrated).parameters
    assert "render_pdf" not in signature(integrated._write_outputs).parameters
    assert "render_pdf" not in signature(
        automatic.run_automatic_demonstration
    ).parameters
    assert write_pdf_report.__module__ == "lithiumscope.reporting.service"


def test_latest_local_candidate_is_integrity_verified_without_activation(monkeypatch, tmp_path):
    from lithiumscope.persistence import save_model
    root = tmp_path / "models"
    monkeypatch.setattr(session, "MODELS_DIR", root)
    monkeypatch.setattr(save_model, "MODELS_DIR", root)
    monkeypatch.setattr(session, "load_latest_model",
                        lambda group: pytest.fail("No active model needed"))
    old, _ = save_model.save_model_bundle("model_1", "old", {"algorithm": "old"}, {},
                                       run_id="training_old")
    new, _ = save_model.save_model_bundle("model_1", "new", {"algorithm": "new"}, {},
                                       run_id="training_new")
    from lithiumscope.core.hashing import file_sha256
    for path, date in ((old, "2026-01-01T00:00:00+00:00"), (new, "2026-01-02T00:00:00+00:00")):
        metadata = path.with_name("metadata.json")
        value = json.loads(metadata.read_text())
        value["saved_at_utc"] = date
        metadata.write_text(json.dumps(value), encoding="utf-8")
        integrity = path.with_name("artifact_integrity.json")
        value = json.loads(integrity.read_text())
        value["files"]["metadata.json"] = file_sha256(metadata)
        integrity.write_text(json.dumps(value), encoding="utf-8")
    before = new.with_name("candidate_status.json").read_bytes()
    bundle, path, info = session.load_model_with_identity("model_1", candidate_run_id="latest_local_candidate")
    assert path == new and bundle["algorithm"] == "new"
    assert info["selection"] == "experimental_demonstration"
    assert new.with_name("candidate_status.json").read_bytes() == before
    assert not (root / "active_models.json").exists()
    new.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="Integridad"):
        session.load_model_with_identity("model_1", candidate_run_id="latest_local_candidate")


def test_latest_local_candidate_does_not_fallback_to_release(monkeypatch, tmp_path):
    monkeypatch.setattr(session, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(session, "load_latest_model", lambda group: pytest.fail("No fallback"))
    with pytest.raises(FileNotFoundError, match="entrene primero"):
        session.load_model_with_identity("model_2", candidate_run_id="latest_local_candidate")
