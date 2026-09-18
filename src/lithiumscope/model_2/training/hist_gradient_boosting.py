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
