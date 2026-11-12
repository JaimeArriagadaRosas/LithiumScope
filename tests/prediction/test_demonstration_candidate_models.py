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
from lithiumscope.prediction import integrated, session


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
    monkeypatch.setattr(predict_command, "choose", lambda *a, **kw: "4")
    monkeypatch.setattr(
        integrated, "run_automatic_demonstration",
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
        integrated.run_automatic_demonstration
    ).parameters
    assert integrated.write_pdf_report.__module__.startswith(
        "lithiumscope.prediction"
    )
