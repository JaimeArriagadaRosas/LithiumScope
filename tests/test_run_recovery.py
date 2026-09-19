import json

from lithiumscope.core.run_resume import recover_abandoned_runs


def test_legacy_running_run_is_marked_crashed(tmp_path):
    run_dir = tmp_path / "model_1" / "runs" / "training_20260919_120000_m0300"
    run_dir.mkdir(parents=True)
    run_file = run_dir / "run.json"
    run_file.write_text(
        json.dumps(
            {
                "run_id": run_dir.name,
                "model_group": "model_1",
                "state": "running",
                "runtime": {},
                "summary": {"training_signature": "abc"},
                "events": [],
                "completed_at_utc": None,
            }
        ),
        encoding="utf-8",
    )

    recovered = recover_abandoned_runs(tmp_path)
    payload = json.loads(run_file.read_text(encoding="utf-8"))

    assert recovered == [run_dir]
    assert payload["state"] == "crashed"
    assert payload["summary"]["crash_reason"]
    assert payload["events"][-1]["event"] == "run_recovered_as_crashed"
