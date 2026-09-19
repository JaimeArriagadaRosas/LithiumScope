from sklearn.ensemble import HistGradientBoostingClassifier


def create_model(random_seed: int = 42, **params):
    defaults = {
        "learning_rate": 0.06,
        "max_iter": 350,
        "max_leaf_nodes": 31,
        "l2_regularization": 0.1,
        "random_state": random_seed,
    }
    defaults.update(params)
    return HistGradientBoostingClassifier(**defaults)


def optuna_space(trial) -> dict:
    return {
        "learning_rate": trial.suggest_float(
            "learning_rate",
            0.01,
            0.2,
            log=True,
        ),
        "max_iter": trial.suggest_int("max_iter", 150, 600),
        "max_leaf_nodes": trial.suggest_int(
            "max_leaf_nodes",
            15,
            63,
        ),
        "min_samples_leaf": trial.suggest_int(
            "min_samples_leaf",
            10,
            50,
        ),
        "l2_regularization": trial.suggest_float(
            "l2_regularization",
            1e-3,
            10.0,
            log=True,
        ),
    }
