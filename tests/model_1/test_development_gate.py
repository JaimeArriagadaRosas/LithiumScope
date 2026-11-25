from types import SimpleNamespace

import numpy as np
import pandas as pd

from lithiumscope.model_1.evaluation.baseline import evaluate_baselines
from lithiumscope.model_1.evaluation.development_gate import apply_development_gate


def test_development_gate_checks_high_tail_and_preserves_relative_order():
    y = pd.Series([1., 100., 2., 90., 3., 80.])
    groups = pd.Series(["a", "a", "b", "b", "c", "c"])
    splits = [(np.flatnonzero(groups != g), np.flatnonzero(groups == g)) for g in ("a", "b", "c")]
    p = SimpleNamespace(y=y, x=pd.DataFrame({"x": y}), groups=groups,
                        frame=pd.DataFrame({"source_dataset": ["A"]*6}), outer_splits=splits,
                        validation_strategy="group_kfold_spatial")
    baseline = evaluate_baselines(p, {"validation": {"prefer_spatial_groups": True, "outer_folds": 3, "random_seed": 42}})
    ranking = pd.DataFrame({"algorithm": ["bad_tail", "good", "missing"],
                            "scientific_candidate_eligible": [True, True, True]})
    results = {"good": (SimpleNamespace(predictions=y.to_numpy(), outer_splits=splits), p),
               "bad_tail": (SimpleNamespace(predictions=np.array([1., 1., 2., 1., 3., 1.]), outer_splits=splits), p)}
    policy = {"max_worst_fold_rmse_ratio": 1., "min_fold_win_fraction": .6,
              "max_high_mae_ratio": 1., "max_high_abs_bias_ratio": 1., "max_source_abs_bias_ratio": 1.}
    out = apply_development_gate(ranking, p, results, baseline[2], policy)
    assert out.algorithm.tolist() == ranking.algorithm.tolist()
    assert out.development_gate_pass.tolist() == [False, True, False]
    assert "high_mae" in out.development_gate_reason.iloc[0]
    assert out.development_gate_reason.iloc[2] == "candidate_evidence_missing"
