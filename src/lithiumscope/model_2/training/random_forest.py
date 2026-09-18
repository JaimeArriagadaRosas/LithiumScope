from sklearn.ensemble import RandomForestClassifier


def create_model(random_seed: int = 42, **params):
    defaults = {
        "n_estimators": 450,
        "class_weight": "balanced",
        "random_state": random_seed,
        "n_jobs": -1,
    }
    defaults.update(params)
    return RandomForestClassifier(**defaults)
