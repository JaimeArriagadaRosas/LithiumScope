"""Candidate evaluation and promotion for trained model artifacts."""

from lithiumscope.prediction.candidates.evaluator import (
    CandidateEvaluationResult,
    evaluate_pending_candidates,
)

__all__ = [
    "CandidateEvaluationResult",
    "evaluate_pending_candidates",
]
