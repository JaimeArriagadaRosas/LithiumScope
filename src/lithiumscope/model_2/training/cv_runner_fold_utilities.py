"""Isolated responsibility with unchanged original API."""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lithiumscope.model_2.training.cv_runner import DeviceInfo, np, pd


def _splitter(
    y: pd.Series,
    groups: pd.Series | None,
    folds: int,
    seed: int,
    *,
    announce: bool = False,
):
    from lithiumscope.model_2.training.cv_runner import StratifiedGroupKFold, StratifiedKFold, logger, pd
    if groups is not None and groups.nunique() >= folds:
        if announce:
            print(
                "    Validación espacial agrupada: "
                f"{groups.nunique()} grupos"
            )
        logger.info(
            "Model 2 validation: StratifiedGroupKFold with %d spatial groups",
            groups.nunique(),
        )
        return (
            StratifiedGroupKFold(
                n_splits=folds,
                shuffle=True,
                random_state=seed,
            ),
            groups,
        )

    if announce:
        print(
            "    Aviso: sin grupos espaciales suficientes; "
            "se usará StratifiedKFold."
        )
    logger.warning(
        "Model 2 spatial groups unavailable/insufficient; "
        "falling back to StratifiedKFold"
    )
    return (
        StratifiedKFold(
            n_splits=folds,
            shuffle=True,
            random_state=seed,
        ),
        None,
    )


def _materialize_splits(
    x,
    y,
    groups,
    folds: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    from lithiumscope.model_2.training.cv_runner import _splitter, np
    splitter, split_groups = _splitter(
        y,
        groups,
        folds,
        seed,
        announce=False,
    )
    iterator = (
        splitter.split(x, y, split_groups)
        if split_groups is not None
        else splitter.split(x, y)
    )
    return [
        (
            np.asarray(train_idx, dtype=int),
            np.asarray(valid_idx, dtype=int),
        )
        for train_idx, valid_idx in iterator
    ]


def _fit_estimator(
    algorithm: str,
    device: DeviceInfo,
    seed: int,
    params: dict,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    groups_train: pd.Series | None,
    inner_folds: int,
    *,
    strict_spatial: bool = False,
):
    from lithiumscope.model_2.training.cv_runner import DeviceInfo, GroupKFold, _materialize_splits, create_model, np, pd
    calibration_cv = None
    if algorithm in {"svm_rbf", "random_forest_spatial_calibrated"}:
        if strict_spatial:
            if groups_train is None or groups_train.nunique() < inner_folds:
                raise ValueError("Spatial calibration requires sufficient groups.")
            calibration_cv = list(GroupKFold(n_splits=inner_folds).split(
                x_train, groups=groups_train,
            ))
            for train_idx, valid_idx in calibration_cv:
                if (
                    np.unique(np.asarray(y_train)[train_idx]).size < 2
                    or np.unique(np.asarray(y_train)[valid_idx]).size < 2
                ):
                    raise ValueError(
                        "Spatial calibration fold lacks both classes; "
                        "no random fallback allowed."
                    )
        else:
            calibration_cv = _materialize_splits(
                x_train, y_train, groups_train, inner_folds, seed + 1000,
            )

    estimator = create_model(
        algorithm,
        device,
        seed,
        params,
        calibrated=True,
        calibration_cv=calibration_cv,
    )
    estimator.fit(x_train, y_train)
    return estimator
