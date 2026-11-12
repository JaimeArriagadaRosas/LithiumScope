"""Versioned RF tuning policies that do not mutate old Optuna studies.

The profile is part of config/model_2.yaml, which contributes to the run
signature. Never change a persisted study's categorical distributions in place.
"""
from __future__ import annotations

from lithiumscope.model_2.training import random_forest


def parameter_space_for(algorithm: str, spec, optimization: dict):
    profile = str(optimization.get("random_forest_search_profile", "legacy"))
    if profile not in {"legacy", "regularized_v2"}:
        raise ValueError(f"Unknown model_2 random forest search profile: {profile}")
    if algorithm == "random_forest" and profile == "regularized_v2":
        return random_forest.optuna_space_regularized
    return spec.module.optuna_space
