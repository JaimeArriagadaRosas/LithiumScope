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



def optuna_space_regularized(trial) -> dict:
    """Conservative RF spectral search for a new, versioned experiment.

    At 783 observations and 126 correlated spectral features, allow at least
    six training observations per leaf and cap depth at twelve. This search
    includes the depth-eight/leaf-ten profile from the spatial pilot.
    It does not alter previous Optuna checkpoints or trained bundles.
    """
    return {
        "n_estimators": trial.suggest_int("n_estimators", 250, 650),
        "max_depth": trial.suggest_categorical(
            "max_depth", [6, 8, 10, 12]
        ),
        "min_samples_leaf": trial.suggest_int(
            "min_samples_leaf", 6, 16
        ),
        "max_features": trial.suggest_categorical(
            "max_features", ["sqrt", "log2", 0.5]
        ),
    }


def optuna_space_regularized_v3(trial) -> dict:
    """Diagnose balanced-class false positives without preselecting RF."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 250, 650),
        "max_depth": trial.suggest_categorical(
            "max_depth", [6, 8, 10, 12]
        ),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 6, 16),
        "max_features": trial.suggest_categorical(
            "max_features", ["sqrt", "log2", 0.5]
        ),
        "class_weight": trial.suggest_categorical(
            "class_weight", [None, "balanced"]
        ),
    }
