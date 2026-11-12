"""Versioned RF search: 126 spectral predictors, grouped CV, no study drift."""
from __future__ import annotations

import json
from pathlib import Path
import pytest

from lithiumscope.core.config import load_config
from lithiumscope.model_2.training import random_forest
from lithiumscope.model_2.training.factory import get_algorithm
from lithiumscope.model_2.training.search_policy import parameter_space_for


class ProbeTrial:
    def __init__(self):
        self.calls = []

    def suggest_int(self, name, low, high):
        self.calls.append((name, low, high))
        if name == "min_samples_leaf":
            return 10
        return 354

    def suggest_categorical(self, name, choices):
        self.calls.append((name, tuple(choices)))
        return 8 if name == "max_depth" else "log2"


def test_regularized_profile_permits_pilot_while_restricting_complexity():
    trial = ProbeTrial()
    spec = get_algorithm("random_forest")
    cfg = load_config("model_2")["optimization"]
    assert cfg["random_forest_search_profile"] == "regularized_v2"
    chosen = parameter_space_for("random_forest", spec, cfg)
    assert chosen is random_forest.optuna_space_regularized
    space = chosen(trial)
    assert space == {
        "n_estimators": 354,
        "max_depth": 8,
        "min_samples_leaf": 10,
        "max_features": "log2",
    }
    assert ("min_samples_leaf", 6, 16) in trial.calls
    assert ("max_depth", (6, 8, 10, 12)) in trial.calls
    assert cfg["per_algorithm_trials"]["random_forest"] >= 12


def test_legacy_study_distribution_remains_unchanged():
    spec = get_algorithm("random_forest")
    legacy = parameter_space_for(
        "random_forest", spec, {"random_forest_search_profile": "legacy"}
    )
    assert legacy is random_forest.optuna_space
    trial = ProbeTrial()
    legacy(trial)
    assert ("min_samples_leaf", 1, 8) in trial.calls
    assert ("max_depth", (None, 8, 12, 16, 24)) in trial.calls


def test_other_five_algorithms_preserve_search_distributions():
    cfg = {"random_forest_search_profile": "regularized_v2"}
    for name in (
        "extra_trees", "hist_gradient_boosting", "xgboost",
        "catboost", "svm_rbf",
    ):
        spec = get_algorithm(name)
        assert parameter_space_for(name, spec, cfg) is spec.module.optuna_space


def test_unknown_profile_fails_without_tuning():
    with pytest.raises(ValueError, match="Unknown"):
        parameter_space_for(
            "random_forest", get_algorithm("random_forest"),
            {"random_forest_search_profile": "unversioned"},
        )


def test_legacy_size_exceptions_have_been_retired():
    root = Path(__file__).resolve().parents[2]
    assert json.loads(
        (root / "config/code_size_legacy_exceptions.json").read_text(
            encoding="utf-8"
        )
    ) == {}
