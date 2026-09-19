from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import warnings

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split

from lithiumscope.core.logger import get_logger

logger = get_logger("model_1.tabnet")
_SEEN_WARNINGS: set[str] = set()


def _log_warnings(records) -> None:
    for record in records:
        message = str(record.message).strip()
        if not message or message in _SEEN_WARNINGS:
            continue
        _SEEN_WARNINGS.add(message)
        logger.warning(
            "TabNet warning captured once: %s",
            message,
        )


def _fit_quietly(model, *args, **kwargs) -> str:
    """Capture library chatter without hiding Python exceptions."""
    output = StringIO()
    with redirect_stdout(output), redirect_stderr(output):
        model.fit(*args, **kwargs)
    captured = output.getvalue().strip()
    if captured:
        lines = [
            line.strip()
            for line in captured.splitlines()
            if line.strip()
        ]
        logger.debug(
            "TabNet internal output captured (%d lines): %s",
            len(lines),
            " | ".join(lines[-5:]),
        )
    return captured


class TabNetRegressorAdapter(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        seed: int = 42,
        max_epochs: int = 150,
        patience: int = 20,
        device_name: str = "auto",
        validation_fraction: float = 0.15,
        refit_full: bool = False,
    ):
        self.seed = seed
        self.max_epochs = max_epochs
        self.patience = patience
        self.device_name = device_name
        self.validation_fraction = validation_fraction
        self.refit_full = refit_full
        self.model_ = None
        self.best_epoch_ = None

    def _new_model(self):
        try:
            from pytorch_tabnet.tab_model import TabNetRegressor
        except ImportError as exc:
            raise RuntimeError(
                "TabNet is optional. Install with: "
                "pip install -e '.[ml]'"
            ) from exc
        return TabNetRegressor(
            seed=self.seed,
            device_name=self.device_name,
            verbose=0,
        )

    def _fit_with_internal_validation(self, x, y):
        if (
            len(x) < 20
            or not 0.0 < self.validation_fraction < 0.5
        ):
            model = self._new_model()
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                _fit_quietly(
                    model,
                    x,
                    y,
                    max_epochs=self.max_epochs,
                    patience=0,
                    batch_size=min(
                        256,
                        max(32, len(x)),
                    ),
                    virtual_batch_size=min(
                        128,
                        max(16, len(x) // 4),
                    ),
                )
            _log_warnings(captured)
            return model, self.max_epochs

        x_train, x_valid, y_train, y_valid = train_test_split(
            x,
            y,
            test_size=self.validation_fraction,
            random_state=self.seed,
        )
        model = self._new_model()
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            _fit_quietly(
                model,
                x_train,
                y_train,
                eval_set=[(x_valid, y_valid)],
                eval_name=["valid"],
                eval_metric=["rmse"],
                max_epochs=self.max_epochs,
                patience=self.patience,
                batch_size=min(
                    256,
                    max(32, len(x_train)),
                ),
                virtual_batch_size=min(
                    128,
                    max(16, len(x_train) // 4),
                ),
            )
        _log_warnings(captured)
        best_epoch = (
            int(
                getattr(
                    model,
                    "best_epoch",
                    self.max_epochs - 1,
                )
            )
            + 1
        )
        logger.info(
            "TabNet selected best_epoch=%d max_epochs=%d patience=%d",
            best_epoch,
            self.max_epochs,
            self.patience,
        )
        return (
            model,
            max(
                1,
                min(best_epoch, self.max_epochs),
            ),
        )

    def fit(self, x, y):
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(
            y,
            dtype=np.float32,
        ).reshape(-1, 1)

        model, best_epoch = (
            self._fit_with_internal_validation(x, y)
        )
        self.best_epoch_ = best_epoch

        if self.refit_full:
            model = self._new_model()
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                _fit_quietly(
                    model,
                    x,
                    y,
                    max_epochs=best_epoch,
                    patience=0,
                    batch_size=min(
                        256,
                        max(32, len(x)),
                    ),
                    virtual_batch_size=min(
                        128,
                        max(16, len(x) // 4),
                    ),
                )
            _log_warnings(captured)

        self.model_ = model
        return self

    def predict(self, x):
        if self.model_ is None:
            raise RuntimeError(
                "TabNet model is not fitted."
            )
        return self.model_.predict(
            np.asarray(x, dtype=np.float32)
        ).reshape(-1)


def create_model(
    random_seed: int = 42,
    device_name: str = "auto",
    final_fit: bool = False,
    **params,
):
    defaults = {
        "seed": random_seed,
        "device_name": device_name,
        "refit_full": final_fit,
    }
    defaults.update(params)
    return TabNetRegressorAdapter(**defaults)


def optuna_space(trial) -> dict:
    return {
        "max_epochs": trial.suggest_int(
            "max_epochs",
            80,
            250,
        ),
        "patience": trial.suggest_int(
            "patience",
            10,
            35,
        ),
    }
