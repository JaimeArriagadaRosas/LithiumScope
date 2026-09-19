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
