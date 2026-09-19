from sklearn.ensemble import RandomForestClassifier

from lithiumscope.core.resources import cpu_worker_budget


def create_model(random_seed: int = 42, **params):
    defaults = {
        "n_estimators": 450,
        "class_weight": "balanced",
        "random_state": random_seed,
        "n_jobs": cpu_worker_budget(),
    }
    defaults.update(params)
    return RandomForestClassifier(**defaults)


def optuna_space(trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 250, 800),
        "max_depth": trial.suggest_categorical(
            "max_depth",
            [None, 8, 12, 16, 24],
        ),
        "min_samples_leaf": trial.suggest_int(
            "min_samples_leaf",
            1,
            8,
        ),
        "max_features": trial.suggest_categorical(
            "max_features",
            ["sqrt", "log2", 0.5, 1.0],
        ),
    }
