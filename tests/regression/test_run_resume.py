import json

from lithiumscope.core.experiment_tracker import ExperimentTracker
from lithiumscope.core.states import RunState


def test_tracker_can_be_reopened_without_overwriting_history(tmp_path):
    tracker = ExperimentTracker(tmp_path, "training_20260919_120000_m0300", "model_1")
    tracker.attach_summary(training_signature="abc")
    tracker.set_state(RunState.CANCELLED)
    tracker.event("algorithm_completed", algorithm="random_forest")

    reopened = ExperimentTracker.load(tmp_path)

    assert reopened.state == RunState.CANCELLED
    assert reopened.payload["summary"]["training_signature"] == "abc"
    assert reopened.payload["events"][-1]["algorithm"] == "random_forest"

    original = json.loads((tmp_path / "run.json").read_text(encoding="utf-8"))
    assert original["run_id"] == "training_20260919_120000_m0300"


def test_crashed_state_is_terminal_and_persisted(tmp_path):
    tracker = ExperimentTracker(tmp_path, "training_20260919_120000_m0300", "model_1")
    tracker.set_state(RunState.CRASHED, crash_reason="native process termination")

    reopened = ExperimentTracker.load(tmp_path)

    assert reopened.state == RunState.CRASHED
    assert reopened.payload["completed_at_utc"] is not None
    assert reopened.payload["summary"]["crash_reason"] == "native process termination"
