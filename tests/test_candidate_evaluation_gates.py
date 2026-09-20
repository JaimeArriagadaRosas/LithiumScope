from lithiumscope.prediction.candidates.gates import (
    model_1_gate,
    model_2_gate,
)


def test_m1_candidate_gate_requires_non_degradation_and_improvement():
    passed, reasons = model_1_gate(
        {
            "rmse": 5.0,
            "mae": 4.0,
            "r2": 0.4,
        },
        {
            "rmse": 4.8,
            "mae": 3.9,
            "r2": 0.45,
        },
    )

    assert passed is True
    assert reasons == []


def test_m2_candidate_gate_allows_threshold_improvement_without_ranking_loss():
    passed, reasons = model_2_gate(
        {
            "roc_auc": 0.625,
            "average_precision": 0.625,
            "balanced_accuracy": 0.5,
        },
        {
            "roc_auc": 0.625,
            "average_precision": 0.625,
            "balanced_accuracy": 0.7,
        },
        same_reference_threshold=True,
    )

    assert passed is True
    assert reasons == []
