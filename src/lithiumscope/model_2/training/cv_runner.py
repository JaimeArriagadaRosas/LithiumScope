from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from lithiumscope.core.checkpoints import FoldCheckpointStore
from lithiumscope.core.device import DeviceInfo
from lithiumscope.core.logger import get_logger
from lithiumscope.core.scientific_checks import (
    assert_group_isolation,
    assert_oof_complete,
)
from lithiumscope.model_2.evaluation.metrics import classification_metrics
from lithiumscope.model_2.training.factory import create_model, get_label
from lithiumscope.runtime.console_status import Spinner

logger = get_logger("model_2.cv_runner")


@dataclass
class ClassificationCVResult:
    algorithm: str
    label: str
    probabilities: np.ndarray
    fold_table: pd.DataFrame
    overall_metrics: dict[str, float]
    elapsed_seconds: float


def _splitter(
    y: pd.Series,
    groups: pd.Series | None,
    folds: int,
    seed: int,
):
    if groups is not None and groups.nunique() >= folds:
        logger.info(
            "Model 2 validation: StratifiedGroupKFold with %d spatial groups",
            groups.nunique(),
        )
        print(f"    Validación espacial agrupada: {groups.nunique()} grupos")
        return StratifiedGroupKFold(
            n_splits=folds,
            shuffle=True,
            random_state=seed,
        ), groups

    logger.warning(
        "Model 2 spatial groups unavailable/insufficient; "
        "falling back to StratifiedKFold"
    )
    print(
        "    Aviso: sin grupos espaciales suficientes; "
        "se usará StratifiedKFold."
    )
    return StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=seed,
    ), None


def run_classification_cv(
    x: pd.DataFrame,
    y: pd.Series,
    algorithm: str,
    device: DeviceInfo,
    seed: int,
    folds: int,
    groups: pd.Series | None = None,
    checkpoint_root: Path | None = None,
) -> ClassificationCVResult:
    splitter, split_groups = _splitter(
        y,
        groups,
        folds,
        seed,
    )
    probabilities = np.full(len(y), np.nan, dtype=float)
    records: list[dict] = []
    started = time.perf_counter()
    label = get_label(algorithm)
    checkpoint = (
        FoldCheckpointStore(checkpoint_root, algorithm)
        if checkpoint_root is not None
        else None
    )
    print(f"\n  ▶ {label}")

    split_iter = (
        splitter.split(x, y, split_groups)
        if split_groups is not None
        else splitter.split(x, y)
    )
    for fold, (train_idx, test_idx) in enumerate(
        split_iter,
        start=1,
    ):
        if split_groups is not None:
            assert_group_isolation(
                train_idx,
                test_idx,
                split_groups,
            )

        cached = checkpoint.load_fold(fold, test_idx) if checkpoint else None
        if cached is not None:
            fold_probabilities = np.asarray(
                cached["predictions"],
                dtype=float,
            )
            probabilities[test_idx] = fold_probabilities
            metrics = dict(cached["metrics"])
            records.append(
                {
                    "fold": fold,
                    **metrics,
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                    "resumed": True,
                }
            )
            print(
                f"      Fold {fold}/{folds} ↻ reutilizado | "
                f"ROC-AUC={metrics.get('roc_auc', float('nan')):.4f} | "
                f"AP={metrics['average_precision']:.4f}"
            )
            continue

        spinner = Spinner(
            f"{label} · fold {fold}/{folds} · ajustando"
        ).start()
        try:
            estimator = create_model(
                algorithm,
                device,
                seed,
            )
            estimator.fit(
                x.iloc[train_idx],
                y.iloc[train_idx],
            )
            fold_probabilities = estimator.predict_proba(
                x.iloc[test_idx]
            )[:, 1]
            probabilities[test_idx] = fold_probabilities
            metrics = classification_metrics(
                y.iloc[test_idx].to_numpy(),
                fold_probabilities,
            )
            records.append(
                {
                    "fold": fold,
                    **metrics,
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                    "resumed": False,
                }
            )
            if checkpoint:
                checkpoint.save_fold(
                    fold,
                    test_idx,
                    fold_probabilities,
                    metrics,
                )
            spinner.succeed(
                f"{label} · fold {fold}/{folds} | "
                f"ROC-AUC={metrics.get('roc_auc', float('nan')):.4f} | "
                f"AP={metrics['average_precision']:.4f} | F1={metrics['f1']:.4f}"
            )
        except BaseException:
            spinner.fail(
                f"{label} · fold {fold}/{folds} interrumpido"
            )
            raise

    assert_oof_complete(probabilities, len(y))
    overall = classification_metrics(
        y.to_numpy(),
        probabilities,
    )
    elapsed = time.perf_counter() - started
    if checkpoint:
        checkpoint.save_algorithm(
            {
                "algorithm": algorithm,
                "label": label,
                "overall_metrics": overall,
                "elapsed_seconds": elapsed,
            }
        )

    return ClassificationCVResult(
        algorithm=algorithm,
        label=label,
        probabilities=probabilities,
        fold_table=pd.DataFrame(records),
        overall_metrics=overall,
        elapsed_seconds=elapsed,
    )
