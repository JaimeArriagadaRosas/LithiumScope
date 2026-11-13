"""Extracted responsibility; original public entrypoint remains stable."""
from __future__ import annotations


def _evaluate_outer_classification_folds(algorithm, checkpoint, device, fixed_threshold_ppm, folds, groups, inner_folds, label, label_quantile, lithium_values, optimize, probabilities, pruning, quality_training_mask, records, scientific_folds, seed, spec, split_groups, split_iter, trials, x, y, parameter_space, calibration_mode='legacy'):
    from lithiumscope.model_2.training.cv_runner import GroupKFold, Spinner, _fit_estimator, _materialize_splits, assert_group_isolation, classification_metrics, create_model, np, optimize_classifier_with_optuna, pd
    for fold, (train_idx, test_idx) in enumerate(split_iter, start=1):
        train_idx = np.asarray(train_idx, dtype=int)
        test_idx = np.asarray(test_idx, dtype=int)
        original_train_count = len(train_idx)
        scientific_fold = (
            scientific_folds[fold - 1] if scientific_folds is not None else None
        )
        if split_groups is not None:
            assert_group_isolation(
                train_idx,
                test_idx,
                split_groups,
            )
        if quality_training_mask is not None:
            # Never remove any test observation. This affects all algorithms,
            # including Optuna inner folds and calibration.
            train_idx = train_idx[quality_training_mask[train_idx]]
            if len(train_idx) < inner_folds or not len(np.unique(
                np.asarray(lithium_values, dtype=float)[train_idx]
                if scientific_fold is not None else y.iloc[train_idx]
            )) > 1:
                raise ValueError("Insufficient two-class QA eligible TRAIN examples")

        cached = (
            checkpoint.load_fold(fold, test_idx)
            if checkpoint
            else None
        )
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
                    **({"label_threshold_ppm": scientific_fold.label_threshold_ppm,
                        "group_overlap": 0,
                        "n_groups_train": scientific_fold.n_groups_train,
                        "n_groups_test": scientific_fold.n_groups_test}
                       if scientific_fold is not None else {}),
                }
            )
            print(
                f"      Fold {fold}/{folds} ↻ reutilizado | "
                f"ROC-AUC={metrics.get('roc_auc', float('nan')):.4f} | "
                f"AP={metrics['average_precision']:.4f}"
            )
            continue

        spinner = Spinner(
            f"{label} · fold {fold}/{folds} · preparando"
        ).start()
        try:
            x_train = x.iloc[train_idx]
            y_train = (
                pd.Series(
                    (np.asarray(lithium_values, dtype=float)[train_idx]
                     >= scientific_fold.label_threshold_ppm).astype(int)
                )
                if scientific_fold is not None else y.iloc[train_idx]
            )
            x_test = x.iloc[test_idx]
            y_test = (
                pd.Series(scientific_fold.y_test)
                if scientific_fold is not None else y.iloc[test_idx]
            )
            groups_train = (
                groups.iloc[train_idx]
                if groups is not None
                else None
            )

            params: dict = {}
            if optimize:
                if scientific_fold is not None:
                    if groups_train is None or groups_train.nunique() < inner_folds:
                        raise ValueError("Insufficient spatial groups for inner CV.")
                    inner_splits = list(GroupKFold(n_splits=inner_folds).split(
                        x_train, groups=groups_train,
                    ))
                else:
                    inner_splits = _materialize_splits(
                        x_train, y_train, groups_train, inner_folds, seed + fold,
                    )

                def factory(candidate: dict):
                    return create_model(
                        algorithm,
                        device,
                        seed,
                        candidate,
                        calibrated=False,
                    )

                def progress(done: int, total: int) -> None:
                    spinner.update(
                        f"{label} · fold {fold}/{folds} · "
                        f"Optuna {done}/{total}"
                    )

                params = optimize_classifier_with_optuna(
                    factory,
                    parameter_space,
                    x_train,
                    y_train,
                    inner_splits,
                    n_trials=trials,
                    random_seed=seed + fold,
                    progress_callback=progress,
                    storage_path=(
                        checkpoint.root
                        / f"optuna_fold_{fold:02d}.db"
                        if checkpoint
                        else None
                    ),
                    study_name=f"{algorithm}_fold_{fold:02d}",
                    pruning_enabled=bool(
                        pruning.get("enabled", True)
                    ),
                    startup_trials=int(
                        pruning.get("startup_trials", 3)
                    ),
                    warmup_folds=int(pruning.get("warmup_folds", 2)),
                    lithium_values=(
                        np.asarray(lithium_values, dtype=float)[train_idx]
                        if scientific_fold is not None else None
                    ),
                    label_quantile=label_quantile,
                    fixed_threshold_ppm=fixed_threshold_ppm,
                )

            spinner.update(
                f"{label} · fold {fold}/{folds} · ajustando"
            )
            if calibration_mode == 'spatial_sigmoid':
                from lithiumscope.model_2.training.calibration_policy import (
                    fit_spatially_calibrated,
                )
                estimator = fit_spatially_calibrated(
                    algorithm, device, seed, params, x_train, y_train,
                    groups_train, inner_folds,
                )
            else:
                estimator = _fit_estimator(
                    algorithm, device, seed, params, x_train, y_train,
                    groups_train, inner_folds,
                    strict_spatial=scientific_fold is not None,
                )
            fold_probabilities = estimator.predict_proba(
                x_test
            )[:, 1]
            probabilities[test_idx] = fold_probabilities
            metrics = classification_metrics(
                y_test.to_numpy(),
                fold_probabilities,
            )
            records.append(
                {
                    "fold": fold,
                    **metrics,
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                    "resumed": False,
                    **({"label_threshold_ppm": scientific_fold.label_threshold_ppm,
                        "group_overlap": 0,
                        "n_groups_train": scientific_fold.n_groups_train,
                        "n_groups_test": scientific_fold.n_groups_test}
                       if scientific_fold is not None else {}),
                }
            )
            if checkpoint:
                checkpoint.save_fold(
                    fold,
                    test_idx,
                    fold_probabilities,
                    metrics,
                    params=params,
                )
            spinner.succeed(
                f"{label} · fold {fold}/{folds} | "
                f"ROC-AUC={metrics.get('roc_auc', float('nan')):.4f} | "
                f"AP={metrics['average_precision']:.4f} | "
                f"F1={metrics['f1']:.4f}"
            )
        except BaseException:
            spinner.fail(
                f"{label} · fold {fold}/{folds} interrumpido"
            )
            raise
