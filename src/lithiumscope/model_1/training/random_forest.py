from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor

from lithiumscope.core.resources import cpu_worker_budget


def create_model(random_seed: int = 42, n_jobs: int | None = None, **params):
    defaults = {
        "n_estimators": 350,
        "random_state": random_seed,
        "n_jobs": cpu_worker_budget() if n_jobs is None else n_jobs,
    }
    defaults.update(params)
    return RandomForestRegressor(**defaults)


def optuna_space(trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 150, 700),
        "max_depth": trial.suggest_int("max_depth", 4, 24),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 12),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 6),
        "max_features": trial.suggest_categorical(
            "max_features",
            ["sqrt", "log2", 1.0],
        ),
    }
