from __future__ import annotations

from dataclasses import dataclass

from sklearn.calibration import CalibratedClassifierCV

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
    module: object


ALGORITHMS = {
    "random_forest": AlgorithmSpec(
        "random_forest",
        "Random Forest",
        random_forest,
    ),
    "random_forest_spatial_calibrated": AlgorithmSpec(
        "random_forest_spatial_calibrated",
        "Random Forest (spatial calibration)",
        random_forest,
    ),
    "extra_trees": AlgorithmSpec(
        "extra_trees",
        "Extra Trees",
        extra_trees,
    ),
    "hist_gradient_boosting": AlgorithmSpec(
        "hist_gradient_boosting",
        "HistGradientBoosting",
        hist_gradient_boosting,
    ),
    "xgboost": AlgorithmSpec(
        "xgboost",
        "XGBoost",
        xgboost,
    ),
    "catboost": AlgorithmSpec(
        "catboost",
        "CatBoost",
        catboost,
    ),
    "svm_rbf": AlgorithmSpec(
        "svm_rbf",
        "SVM (RBF)",
        svm_rbf,
    ),
}


def get_algorithm(name: str) -> AlgorithmSpec:
    try:
        return ALGORITHMS[name]
    except KeyError as exc:
        raise ValueError(
            f"Algoritmo Modelo 2 no soportado: {name}"
        ) from exc


def get_label(name: str) -> str:
    return get_algorithm(name).label


def create_model(
    name: str,
    device: DeviceInfo,
    seed: int,
    params: dict | None = None,
    *,
    calibrated: bool = True,
    calibration_cv=None,
):
    params = dict(params or {})
    spec = get_algorithm(name)

    if name == "xgboost":
        return xgboost.create_model(
            device,
            seed,
            **params,
        )
    if name == "catboost":
        return catboost.create_model(
            device,
            seed,
            **params,
        )
    if name == "svm_rbf":
        return svm_rbf.create_model(
            seed,
            calibrated=calibrated,
            calibration_cv=calibration_cv,
            **params,
        )
    if name == "random_forest_spatial_calibrated":
        defaults = {
            "n_estimators": 160,
            "max_depth": 9,
            "min_samples_leaf": 4,
        }
        defaults.update(params)
        base = random_forest.create_model(seed, **defaults)
        if not calibrated:
            return base
        if calibration_cv is None:
            raise ValueError(
                "Spatial RF calibration requires explicit group-isolated folds."
            )
        return CalibratedClassifierCV(
            estimator=base,
            method="sigmoid",
            cv=calibration_cv,
            ensemble=True,
        )
    return spec.module.create_model(
        seed,
        **params,
    )
