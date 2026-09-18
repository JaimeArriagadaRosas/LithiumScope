from __future__ import annotations

from dataclasses import dataclass

from lithiumscope.core.device import DeviceInfo
from lithiumscope.model_2.training import (
    catboost,
    extra_trees,
    hist_gradient_boosting,
    random_forest,
    svm_rbf,
    xgboost,
)


@dataclass(frozen=True)
class AlgorithmSpec:
    name: str
    label: str


LABELS = {
    "random_forest": "Random Forest",
    "extra_trees": "Extra Trees",
    "hist_gradient_boosting": "HistGradientBoosting",
    "xgboost": "XGBoost",
    "catboost": "CatBoost",
    "svm_rbf": "SVM (RBF)",
}


def get_label(name: str) -> str:
    return LABELS.get(name, name)


def create_model(name: str, device: DeviceInfo, seed: int):
    if name == "random_forest":
        return random_forest.create_model(seed)
    if name == "extra_trees":
        return extra_trees.create_model(seed)
    if name == "hist_gradient_boosting":
        return hist_gradient_boosting.create_model(seed)
    if name == "xgboost":
        return xgboost.create_model(device, seed)
    if name == "catboost":
        return catboost.create_model(device, seed)
    if name == "svm_rbf":
        return svm_rbf.create_model(seed)
    raise ValueError(f"Algoritmo Modelo 2 no soportado: {name}")
