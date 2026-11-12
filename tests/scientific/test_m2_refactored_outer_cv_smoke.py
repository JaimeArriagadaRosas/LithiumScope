"""Exercise a real, small CPU Random Forest after CV module extraction."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from lithiumscope.model_2.training.cv_runner import run_classification_cv


def test_real_grouped_rf_cv_preserves_oof_and_does_not_use_gpu():
    rng = np.random.default_rng(20261112)
    n = 90
    groups = pd.Series(np.repeat([f"region_{i}" for i in range(6)], 15))
    columns = ["red_mean", "green_mean", "blue_mean",
               "nir_mean", "swir16_mean", "swir22_mean"]
    x = pd.DataFrame(rng.normal(size=(n, 6)), columns=columns)
    y = pd.Series(
        ((x.red_mean + 0.5 * x.nir_mean) > 0.0).astype(int)
    )
    result = run_classification_cv(
        x,
        y,
        "random_forest",
        device=SimpleNamespace(accelerator="cpu"),
        seed=42,
        folds=3,
        groups=groups,
        optimization={
            "enabled": False,
            "random_forest_search_profile": "regularized_v2",
        },
        inner_folds=3,
    )
    assert result.algorithm == "random_forest"
    assert len(result.probabilities) == n
    assert np.isfinite(result.probabilities).all()
    assert ((result.probabilities >= 0) & (result.probabilities <= 1)).all()
    assert len(result.fold_table) == 3
    assert result.overall_metrics["roc_auc"] > 0.5
    assert result.final_params == {}
