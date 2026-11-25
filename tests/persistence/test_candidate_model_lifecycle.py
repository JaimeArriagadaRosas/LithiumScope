import json

import joblib

import lithiumscope.persistence.active_models as active_models
import lithiumscope.persistence.candidates as candidates
import lithiumscope.persistence.save_model as save_model


def test_candidate_save_does_not_replace_active(
    monkeypatch,
    tmp_path,
):
    models = tmp_path / "models"
    active_registry = (
        models / "active_models.json"
    )
    old = (
        models
        / "model_1"
        / "trained"
        / "old"
        / "model.joblib"
    )
    old.parent.mkdir(
        parents=True
    )
    joblib.dump(
        {"algorithm": "old"},
        old,
    )

    monkeypatch.setattr(
        active_models,
        "PROJECT_ROOT",
        tmp_path,
    )
    monkeypatch.setattr(
        active_models,
        "MODELS_DIR",
        models,
    )
    monkeypatch.setattr(
        active_models,
        "ACTIVE_MODELS_PATH",
        active_registry,
    )
    monkeypatch.setattr(
        save_model,
        "MODELS_DIR",
        models,
    )
    monkeypatch.setattr(
        save_model,
        "CONFIG_DIR",
        tmp_path / "config",
    )
    monkeypatch.setattr(
        candidates,
        "MODELS_DIR",
        models,
    )

    active_models.set_active_model(
        "model_1",
        old,
        source={"type": "test"},
    )

    candidate, _ = (
        save_model.save_model_bundle(
            "model_1",
            "winner_test",
            {"algorithm": "new"},
            {
                "algorithm": "new",
                "metrics": {
                    "rmse": 1.0
                },
            },
            run_id=(
                "training_20260920_190000_m0300"
            ),
            activate=False,
        )
    )

    assert (
        active_models.active_model_path(
            "model_1"
        )
        == old
    )
    status = json.loads(
        (
            candidate.parent
            / "candidate_status.json"
        ).read_text(
            encoding="utf-8"
        )
    )
    assert status["status"] == "pending"
    assert (
        candidates.latest_pending_candidate(
            "model_1"
        )
        == candidate
    )


def test_candidate_status_removes_rejected_from_pending(
    monkeypatch,
    tmp_path,
):
    models = tmp_path / "models"
    model_path = (
        models
        / "model_2"
        / "trained"
        / "training_x"
        / "model.joblib"
    )
    model_path.parent.mkdir(
        parents=True
    )
    model_path.write_bytes(b"x")
    (
        model_path.parent
        / "candidate_status.json"
    ).write_text(
        json.dumps(
            {"status": "pending"}
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        candidates,
        "MODELS_DIR",
        models,
    )
    monkeypatch.setattr(
        candidates,
        "active_model_path",
        lambda group: None,
    )

    assert (
        candidates.latest_pending_candidate(
            "model_2"
        )
        == model_path
    )

    candidates.update_candidate_status(
        model_path,
        status="rejected",
        evaluation_id="evaluation_x",
    )

    assert (
        candidates.latest_pending_candidate(
            "model_2"
        )
        is None
    )
