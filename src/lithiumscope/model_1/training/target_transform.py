from __future__ import annotations

import numpy as np
from sklearn.compose import TransformedTargetRegressor


SUPPORTED_TARGET_TRANSFORMS = (
    "identity",
    "log1p",
)


def normalize_target_transform(
    name: str | None,
) -> str:
    value = str(
        name or "identity"
    ).strip().lower()
    if value not in SUPPORTED_TARGET_TRANSFORMS:
        raise ValueError(
            "Transformación de target no soportada: "
            f"{value}. Opciones: "
            + ", ".join(
                SUPPORTED_TARGET_TRANSFORMS
            )
        )
    return value


def validate_target_transform(
    name: str,
    y,
) -> str:
    value = normalize_target_transform(
        name
    )
    if value == "log1p":
        target = np.asarray(
            y,
            dtype=float,
        )
        finite = target[
            np.isfinite(target)
        ]
        if finite.size and float(
            finite.min()
        ) < 0.0:
            raise ValueError(
                "log1p requiere un target no negativo."
            )
    return value


def wrap_target_regressor(
    estimator,
    name: str,
):
    value = normalize_target_transform(
        name
    )
    if value == "identity":
        return estimator
    return TransformedTargetRegressor(
        regressor=estimator,
        func=np.log1p,
        inverse_func=np.expm1,
        check_inverse=False,
    )


def unwrap_fitted_regressor(
    estimator,
):
    fitted = getattr(
        estimator,
        "regressor_",
        None,
    )
    if fitted is not None:
        return fitted
    return getattr(
        estimator,
        "regressor",
        estimator,
    )


def transform_label(
    name: str,
) -> str:
    value = normalize_target_transform(
        name
    )
    return (
        "target original"
        if value == "identity"
        else "target log1p"
    )
