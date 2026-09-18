from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def create_model(random_seed: int = 42, **params):
    defaults = {
        "kernel": "rbf",
        "C": 5.0,
        "gamma": "scale",
        "probability": True,
        "class_weight": "balanced",
        "random_state": random_seed,
    }
    defaults.update(params)
    return Pipeline([("scaler", StandardScaler()), ("model", SVC(**defaults))])
