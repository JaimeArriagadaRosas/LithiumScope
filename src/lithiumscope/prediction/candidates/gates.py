from __future__ import annotations

import math


def _valid_metrics(metrics, required, *, bounded=False):
    try:
        for key in required:
            value = metrics[key]
            if isinstance(value, bool) or not math.isfinite(value):
                return False
            if bounded and not 0 <= value <= 1:
                return False
            if key in {'rmse', 'mae'} and value < 0:
                return False
            if key == 'r2' and value > 1:
                return False
    except (KeyError, TypeError, ValueError):
        return False
    return True


def model_1_gate(
    active: dict,
    candidate: dict,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    required = (
        "rmse",
        "mae",
        "r2",
    )
    if not all(_valid_metrics(metrics, required) for metrics in (active, candidate)):
        return (
            False,
            ["metricas externas incompletas"],
        )

    checks = {
        "RMSE externo empeoro": (
            candidate["rmse"]
            <= active["rmse"]
        ),
        "MAE externo empeoro": (
            candidate["mae"]
            <= active["mae"]
        ),
        "R2 externo empeoro": (
            candidate["r2"]
            >= active["r2"]
        ),
    }
    for reason, ok in checks.items():
        if not ok:
            reasons.append(reason)

    strict = (
        candidate["rmse"]
        < active["rmse"]
        or candidate["mae"]
        < active["mae"]
        or candidate["r2"]
        > active["r2"]
    )
    if not strict:
        reasons.append(
            "sin mejora externa estricta"
        )

    return (
        all(checks.values())
        and strict,
        reasons,
    )


def model_2_gate(
    active: dict,
    candidate: dict,
    *,
    same_reference_threshold: bool,
) -> tuple[bool, list[str]]:
    if not same_reference_threshold:
        return (
            False,
            [
                "el umbral de Li de referencia cambio; "
                "la comparacion binaria no es equivalente"
            ],
        )

    required = (
        "roc_auc",
        "average_precision",
        "balanced_accuracy",
    )
    if not all(_valid_metrics(metrics, required, bounded=True) for metrics in (active, candidate)):
        return (
            False,
            ["metricas externas incompletas"],
        )

    checks = {
        "ROC-AUC externo empeoro": (
            candidate["roc_auc"]
            >= active["roc_auc"]
        ),
        "Average Precision externo empeoro": (
            candidate["average_precision"]
            >= active["average_precision"]
        ),
        "Balanced Accuracy externa empeoro": (
            candidate["balanced_accuracy"]
            >= active["balanced_accuracy"]
        ),
    }
    reasons = [
        reason
        for reason, ok in checks.items()
        if not ok
    ]
    strict = (
        candidate["roc_auc"]
        > active["roc_auc"]
        or candidate["average_precision"]
        > active["average_precision"]
        or candidate["balanced_accuracy"]
        > active["balanced_accuracy"]
    )
    if not strict:
        reasons.append(
            "sin mejora externa estricta"
        )

    return (
        all(checks.values())
        and strict,
        reasons,
    )


def comparable_cases(active, candidate, expected, *, prediction_column):
    """Do not compare independently filtered populations."""
    import numpy as np
    import pandas as pd

    ids = set(expected["case_id"])
    for frame in (active, candidate, expected):
        if "case_id" not in frame or frame["case_id"].isna().any() or frame["case_id"].duplicated().any():
            return False
    if set(active["case_id"]) != ids or set(candidate["case_id"]) != ids:
        return False
    truth = expected.set_index("case_id")["Li_icpms"]
    for frame in (active, candidate):
        if not {"Li_icpms", prediction_column}.issubset(frame.columns):
            return False
        aligned = frame.set_index("case_id").reindex(truth.index)
        observed = pd.to_numeric(aligned["Li_icpms"], errors="coerce")
        reference = pd.to_numeric(truth, errors="coerce")
        prediction = pd.to_numeric(aligned[prediction_column], errors="coerce")
        if not np.isfinite(reference).all() or not np.isfinite(prediction).all():
            return False
        if not np.array_equal(observed.to_numpy(), reference.to_numpy()):
            return False
    return bool(ids)
