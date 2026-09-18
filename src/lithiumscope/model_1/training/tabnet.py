from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class TabNetRegressorAdapter(BaseEstimator, RegressorMixin):
    def __init__(self, seed: int = 42, max_epochs: int = 150, patience: int = 20, device_name: str = "auto"):
        self.seed = seed
        self.max_epochs = max_epochs
        self.patience = patience
        self.device_name = device_name
        self.model_ = None

    def fit(self, x, y):
        try:
            from pytorch_tabnet.tab_model import TabNetRegressor
        except ImportError as exc:
            raise RuntimeError("TabNet is optional. Install with: pip install -e '.[ml]'") from exc
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        self.model_ = TabNetRegressor(seed=self.seed, device_name=self.device_name, verbose=0)
        self.model_.fit(x, y, max_epochs=self.max_epochs, patience=self.patience,
                        batch_size=min(256, max(32, len(x))),
                        virtual_batch_size=min(128, max(16, len(x) // 4)))
        return self

    def predict(self, x):
        if self.model_ is None:
            raise RuntimeError("TabNet model is not fitted.")
        return self.model_.predict(np.asarray(x, dtype=np.float32)).reshape(-1)


def create_model(random_seed: int = 42, device_name: str = "auto", **params):
    defaults = {"seed": random_seed, "device_name": device_name}
    defaults.update(params)
    return TabNetRegressorAdapter(**defaults)


def optuna_space(trial) -> dict:
    return {"max_epochs": trial.suggest_int("max_epochs", 80, 250), "patience": trial.suggest_int("patience", 10, 35)}
