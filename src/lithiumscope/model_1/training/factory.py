from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType

from sklearn.base import clone
from sklearn.pipeline import Pipeline

from lithiumscope.core.device import DeviceInfo
from lithiumscope.model_1.steps.step_08_preprocessing import build_preprocessor
from lithiumscope.model_1.training import (
    catboost,
    hist_gradient_boosting,
    random_forest,
    svm,
    tabnet,
    xgboost,
)


@dataclass(frozen=True)
class AlgorithmSpec:
    name: str
    label: str
    module: ModuleType
    optional_dependency: str | None = None


ALGORITHMS = {
    "random_forest": AlgorithmSpec("random_forest", "Random Forest", random_forest),
    "xgboost": AlgorithmSpec("xgboost", "XGBoost", xgboost, "xgboost"),
    "svm": AlgorithmSpec("svm", "SVM (RBF)", svm),
    "tabnet": AlgorithmSpec("tabnet", "TabNet", tabnet, "pytorch-tabnet"),
    "hist_gradient_boosting": AlgorithmSpec(
        "hist_gradient_boosting",
        "HistGradientBoosting",
        hist_gradient_boosting,
    ),
    "catboost": AlgorithmSpec("catboost", "CatBoost", catboost, "catboost"),
}


def get_algorithm(name: str) -> AlgorithmSpec:
    try:
        return ALGORITHMS[name]
    except KeyError as exc:
        raise ValueError(f"Algoritmo no soportado: {name}") from exc


def create_model(
    name: str,
    device: DeviceInfo,
    seed: int,
    params: dict,
    *,
    final_fit: bool = False,
):
    spec = get_algorithm(name)
    if name in {"xgboost", "catboost"}:
        return spec.module.create_model(
            device=device,
            random_seed=seed,
            **params,
        )
    if name in {"random_forest", "hist_gradient_boosting"}:
        return spec.module.create_model(random_seed=seed, **params)
    if name == "tabnet":
        backend = "cuda" if device.accelerator == "cuda" else "cpu"
        return spec.module.create_model(
            random_seed=seed,
            device_name=backend,
            final_fit=final_fit,
            **params,
        )
    return spec.module.create_model(**params)


def build_pipeline(
    name: str,
    device: DeviceInfo,
    seed: int,
    schema,
    params: dict,
    *,
    final_fit: bool = False,
):
    preprocessor = build_preprocessor(schema, model_family=name)
    return Pipeline(
        steps=[
            ("preprocess", clone(preprocessor)),
            (
                "model",
                create_model(
                    name,
                    device,
                    seed,
                    params,
                    final_fit=final_fit,
                ),
            ),
        ]
    )
