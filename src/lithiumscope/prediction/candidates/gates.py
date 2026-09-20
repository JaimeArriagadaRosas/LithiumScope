from __future__ import annotations


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
    if any(
        key not in active
        or key not in candidate
        for key in required
    ):
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
    if any(
        key not in active
        or key not in candidate
        for key in required
    ):
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
