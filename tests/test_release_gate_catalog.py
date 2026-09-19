import json

import lithiumscope.results.catalog as catalog_module


def _write_run(root, group, run_id, *, gate):
    run_dir = root / group / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "model_group": group,
                "state": "completed",
                "runtime": {"git_commit": "abc123"},
                "summary": {
                    "winner": "svm_rbf",
                    "dataset_sha256": "deadbeef",
                    "failed_algorithms": [],
                    "release_gate_pass": gate,
                },
            }
        ),
        encoding="utf-8",
    )


def test_model_2_requires_release_gate(monkeypatch, tmp_path):
    _write_run(
        tmp_path,
        "model_2",
        "training_20260919_120000_m0300",
        gate=False,
    )
    monkeypatch.setattr(catalog_module, "RESULTS_DIR", tmp_path)

    frame = catalog_module.run_catalog("model_2")

    assert bool(frame.iloc[0]["release_candidate"]) is False
    assert bool(frame.iloc[0]["release_gate_pass"]) is False


def test_model_1_does_not_require_model_2_gate(monkeypatch, tmp_path):
    _write_run(
        tmp_path,
        "model_1",
        "training_20260919_120000_m0300",
        gate=None,
    )
    monkeypatch.setattr(catalog_module, "RESULTS_DIR", tmp_path)

    frame = catalog_module.run_catalog("model_1")

    assert bool(frame.iloc[0]["release_candidate"]) is True
