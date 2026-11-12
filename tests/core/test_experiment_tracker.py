import json

from lithiumscope.core.experiment_tracker import ExperimentTracker
from lithiumscope.core.states import RunState


def test_experiment_tracker_records_lifecycle(tmp_path):
    tracker = ExperimentTracker(tmp_path, "run-1", "model_1")
    tracker.set_state(RunState.RUNNING)
    tracker.event("algorithm_started", algorithm="rf")
    tracker.set_state(RunState.COMPLETED, winner="rf")

    payload = json.loads((tmp_path / "run.json").read_text(encoding="utf-8"))
    assert payload["state"] == "completed"
    assert payload["summary"]["winner"] == "rf"
    assert payload["events"][0]["algorithm"] == "rf"
