import json

import pytest

from lithiumscope.cli.candidate_admin import main
from lithiumscope.prediction.candidates import evaluator
from lithiumscope.prediction.candidates.gates import model_1_gate, model_2_gate


def test_demo_never_updates_candidate_or_active(tmp_path):
    active = tmp_path / "active" / "model.joblib"
    candidate = tmp_path / "candidate" / "model.joblib"
    for path in (active, candidate):
        path.parent.mkdir()
        path.write_bytes(b"model")
    state = candidate.parent / "candidate_status.json"
    state.write_text(json.dumps({"status": "pending"}))
    original = state.read_bytes()
    decision = evaluator._record_decision(
        model_group="model_1", active_path=active, candidate_path=candidate,
        active_metrics={"rmse": 2}, candidate_metrics={"rmse": 1},
        passed=True, reasons=[], evaluation_id="test", promote_if_better=True,
    )
    assert decision["demo_metrics_passed"]
    assert not decision["passed_gate"]
    assert not decision["promoted"]
    assert decision["status"] == "comparison_only"
    assert state.read_bytes() == original


def test_api_promotion_fails_before_evaluation(monkeypatch):
    def forbidden():
        pytest.fail("must not read evaluation targets")
    monkeypatch.setattr(evaluator, "_evaluation_targets", forbidden)
    with pytest.raises(ValueError, match="independent holdout"):
        evaluator.evaluate_pending_candidates(promote_if_better=True)


def test_cli_promotion_blocked():
    assert main(["promote", "--confirm"]) == 2


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, None])
def test_invalid_m1_metrics_cannot_pass(value):
    active = {"rmse": 2, "mae": 1, "r2": 0.4}
    candidate = {"rmse": value, "mae": 0.5, "r2": 0.5}
    assert not model_1_gate(active, candidate)[0]


def test_impossible_m2_metric_cannot_pass():
    active = {"roc_auc": 0.6, "average_precision": 0.6, "balanced_accuracy": 0.5}
    candidate = dict(active, roc_auc=1.1)
    assert not model_2_gate(active, candidate, same_reference_threshold=True)[0]

def test_comparison_requires_same_cases_and_truth():
    import pandas as pd
    from lithiumscope.prediction.candidates.gates import comparable_cases

    expected = pd.DataFrame({"case_id": ["a", "b"], "Li_icpms": [10, 20]})
    active = expected.assign(score=[0.2, 0.8])
    assert comparable_cases(active, active.iloc[::-1], expected, prediction_column="score")
    assert not comparable_cases(active, active.iloc[:1], expected, prediction_column="score")
    assert not comparable_cases(active, active.assign(score=[0.2, float("nan")]), expected, prediction_column="score")
    assert not comparable_cases(active, active.assign(Li_icpms=[10, 21]), expected, prediction_column="score")
