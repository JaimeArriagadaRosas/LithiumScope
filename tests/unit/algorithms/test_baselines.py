from types import SimpleNamespace

import numpy as np
import pandas as pd

from lithiumscope.model_1.evaluation.baseline import evaluate_mean_baseline
from lithiumscope.model_2.evaluation.baseline import evaluate_prior_baseline


def test_model_1_mean_baseline_produces_complete_oof_predictions():
    prepared = SimpleNamespace(
        x=pd.DataFrame({"x": np.arange(20, dtype=float)}),
        y=pd.Series(np.linspace(1.0, 20.0, 20)),
    )
    config = {
        "validation": {
            "outer_folds": 5,
            "inner_folds": 5,
            "random_seed": 42,
        }
    }

    ranking, folds, predictions = evaluate_mean_baseline(prepared, config)

    assert ranking["is_baseline"] is True
    assert len(folds) == 5
    assert np.isfinite(predictions).all()


def test_model_2_prior_baseline_respects_grouped_cv():
    x = pd.DataFrame({"x": np.arange(40, dtype=float)})
    y = pd.Series([0, 1] * 20)
    groups = pd.Series([f"g{index // 2}" for index in range(40)])

    ranking, folds, probabilities = evaluate_prior_baseline(
        x=x,
        y=y,
        folds=5,
        seed=42,
        groups=groups,
    )

    assert ranking["is_baseline"] is True
    assert len(folds) == 5
    assert np.isfinite(probabilities).all()
