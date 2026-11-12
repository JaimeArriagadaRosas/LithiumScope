import json

import lithiumscope.persistence.active_models as active_models
import lithiumscope.results.catalog as catalog_module


def test_active_release_candidate_requires_active_promoted_run(
    monkeypatch,
    tmp_path,
):
    results = tmp_path / "results"
    models = tmp_path / "models"
    active_registry = (
        models / "active_models.json"
    )
    model_path = (
        models
        / "model_1"
        / "trained"
        / "training_active"
        / "model.joblib"
    )
    model_path.parent.mkdir(
        parents=True
    )
    model_path.write_bytes(b"model")
    run_dir = (
        results
        / "model_1"
        / "runs"
        / "training_active"
    )
    run_dir.mkdir(
        parents=True
    )
    (
        run_dir / "run.json"
    ).write_text(
        json.dumps(
            {
                "run_id": "training_active",
                "model_group": "model_1",
                "state": "completed",
                "runtime": {
                    "git_commit": "abc123"
                },
                "summary": {
                    "winner": "catboost",
                    "dataset_sha256": "dataset",
                    "failed_algorithms": [],
                },
            }
        ),
        encoding="utf-8",
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
        catalog_module,
        "RESULTS_DIR",
        results,
    )
    monkeypatch.setattr(
        catalog_module,
        "active_model_path",
        active_models.active_model_path,
    )

    active_models.set_active_model(
        "model_1",
        model_path,
        source={
            "type": "candidate_evaluation"
        },
    )

    candidate = (
        catalog_module.active_release_candidate(
            "model_1"
        )
    )

    assert candidate is not None
    assert (
        candidate["run_id"]
        == "training_active"
    )
