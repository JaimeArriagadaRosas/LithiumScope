"""Versioned M2 tuning policies, independent across all six competitors.

Profile changes are part of the config/run signature and must never reuse
incompatible existing Optuna study distributions.
"""
from __future__ import annotations

from lithiumscope.model_2.training import extra_trees, random_forest


def parameter_space_for(algorithm: str, spec, optimization: dict):
    rf_profile = str(optimization.get("random_forest_search_profile", "legacy"))
    et_profile = str(optimization.get("extra_trees_search_profile", "legacy"))
    if rf_profile not in {"legacy", "regularized_v2", "regularized_v3"}:
        raise ValueError(f"Unknown model_2 random forest search profile: {rf_profile}")
    if et_profile not in {"legacy", "regularized_v2"}:
        raise ValueError(f"Unknown model_2 extra trees search profile: {et_profile}")
    if algorithm == "random_forest":
        if rf_profile == "regularized_v3":
            return random_forest.optuna_space_regularized_v3
        if rf_profile == "regularized_v2":
            return random_forest.optuna_space_regularized
    if algorithm == "extra_trees" and et_profile == "regularized_v2":
        return extra_trees.optuna_space_regularized_v2
    return spec.module.optuna_space
