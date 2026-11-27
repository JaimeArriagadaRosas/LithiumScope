"""Candidate evaluation and holdout have independent canonical owners."""
from importlib import import_module
from pathlib import Path
import pytest


@pytest.mark.parametrize("name", [
    "lithiumscope.evaluation.candidates.evaluator",
    "lithiumscope.evaluation.candidates.evaluator_evaluate_pending_candidates",
    "lithiumscope.evaluation.candidates.gates",
    "lithiumscope.evaluation.holdout.registry",
    "lithiumscope.evaluation.holdout.scoring",
    "lithiumscope.evaluation.holdout.evaluation",
    "lithiumscope.prediction.execution.model_2_scoring",
])
def test_evaluation_canonical_owner(name):
    assert import_module(name).__name__ == name


def test_no_candidate_compatibility_sources():
    root = Path(__file__).resolve().parents[2] / "src/lithiumscope/prediction"
    assert not list((root / "candidates").rglob("*.py"))
    assert not (root / "candidate_evaluation.py").is_file()
