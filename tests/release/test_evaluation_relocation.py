from importlib import import_module
import pytest

@pytest.mark.parametrize("old,new", [
  [
    "lithiumscope.prediction.candidates.evaluator",
    "lithiumscope.evaluation.candidates.evaluator"
  ],
  [
    "lithiumscope.prediction.candidates.evaluator_evaluate_pending_candidates",
    "lithiumscope.evaluation.candidates.evaluator_evaluate_pending_candidates"
  ],
  [
    "lithiumscope.prediction.candidates.gates",
    "lithiumscope.evaluation.candidates.gates"
  ],
  [
    "lithiumscope.prediction.candidates.scoring",
    "lithiumscope.prediction.execution.candidate_scoring"
  ],
  [
    "lithiumscope.prediction.candidates.holdout.registry",
    "lithiumscope.evaluation.holdout.registry"
  ],
  [
    "lithiumscope.prediction.candidates.holdout.scoring",
    "lithiumscope.evaluation.holdout.scoring"
  ],
  [
    "lithiumscope.prediction.candidates.holdout.evaluation",
    "lithiumscope.evaluation.holdout.evaluation"
  ]
])
def test_evaluation_compatibility(old,new):
    assert import_module(old) is import_module(new)
