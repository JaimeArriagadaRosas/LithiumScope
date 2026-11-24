from __future__ import annotations

from sklearn.svm import SVR


def create_model(**params):
    defaults = {"kernel": "rbf", "C": 10.0, "epsilon": 0.1, "gamma": "scale"}
    from lithiumscope.resources.manager import svm_cache_limit
    defaults.update(params)
    defaults['cache_size']=svm_cache_limit(defaults.get('cache_size',200.))
    return SVR(**defaults)


def optuna_space(trial) -> dict:
    return {
        "C": trial.suggest_float("C", 0.1, 100.0, log=True),
        "epsilon": trial.suggest_float("epsilon", 0.01, 1.0, log=True),
        "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
    }
