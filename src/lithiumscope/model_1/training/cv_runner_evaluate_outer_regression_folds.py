"""Extracted responsibility; original public entrypoint remains stable."""
from __future__ import annotations


def _evaluate_outer_regression_folds(algorithm, checkpoint, device, folds, groups, label, optimize, outer_splits, outer_strategy, predictions, prepared, pruning, require_groups, seed, spec, target_transform, trials, validation):
    from lithiumscope.model_1.training.cv_runner import Spinner, assert_group_isolation, build_pipeline, logger, materialize_regression_splits, np, optimize_with_optuna, regression_metrics
    for fold, (train_idx, test_idx) in enumerate(
        outer_splits,
        start=1,
    ):
        if (
            groups is not None
            and outer_strategy == "group_kfold_spatial"
        ):
            assert_group_isolation(
                train_idx,
                test_idx,
                groups,
            )
        cached = checkpoint.load_fold(fold, test_idx) if checkpoint else None
        if cached is not None:
            fold_predictions = np.asarray(
                cached["predictions"],
                dtype=float,
            )
            predictions[test_idx] = fold_predictions
            metrics = dict(cached["metrics"])
            folds.append(
                {
                    "fold": fold,
                    **metrics,
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                    "validation_strategy": outer_strategy,
                    "resumed": True,
                }
            )
            print(
                f"      Fold externo {fold}/{len(outer_splits)} "
                f"↻ reutilizado | RMSE={metrics['rmse']:.4f} | "
                f"MAE={metrics['mae']:.4f} | R²={metrics['r2']:.4f}"
            )
            continue

        spinner = Spinner(
            f"{label} · fold {fold}/{len(outer_splits)} · preparando"
        ).start()
        try:
            x_train = prepared.x.iloc[train_idx]
            y_train = prepared.y.iloc[train_idx]
            x_test = prepared.x.iloc[test_idx]
            y_test = prepared.y.iloc[test_idx]
            groups_train = (
                groups.iloc[train_idx]
                if groups is not None
                else None
            )

            params: dict = {}
            if optimize:
                def factory(candidate: dict):
                    return build_pipeline(
                        algorithm,
                        device,
                        seed,
                        prepared.schema,
                        candidate,
                        target_transform=target_transform,
                    )

                def progress(done: int, total: int) -> None:
                    spinner.update(
                        f"{label} · fold {fold}/{len(outer_splits)} · "
                        f"Optuna {done}/{total}"
                    )

                inner_splits, _ = materialize_regression_splits(
                    x_train,
                    y_train,
                    groups_train,
                    int(validation["inner_folds"]),
                    seed + fold,
                    require_groups=require_groups,
                )
                params = optimize_with_optuna(
                    factory,
                    spec.module.optuna_space,
                    x_train,
                    y_train,
                    inner_splits,
                    trials,
                    progress_callback=progress,
                    storage_path=(
                        checkpoint.root / f"optuna_fold_{fold:02d}.db"
                        if checkpoint
                        else None
                    ),
                    study_name=f"{algorithm}_fold_{fold:02d}",
                    random_seed=seed + fold,
                    pruning_enabled=bool(
                        pruning.get("enabled", True)
                    ),
                    startup_trials=int(
                        pruning.get("startup_trials", 4)
                    ),
                    warmup_folds=int(
                        pruning.get("warmup_folds", 2)
                    ),
                )

            spinner.update(
                f"{label} · fold {fold}/{len(outer_splits)} · ajustando"
            )
            estimator = build_pipeline(
                algorithm,
                device,
                seed,
                prepared.schema,
                params,
                target_transform=target_transform,
            )
            estimator.fit(x_train, y_train)
            fold_predictions = estimator.predict(x_test)
            predictions[test_idx] = fold_predictions
            metrics = regression_metrics(
                y_test,
                fold_predictions,
            ).to_dict()
            # Paired holdout stress test, never fitted on the outer test set.
            # Noise is scaled from outer-training features only.
            numeric = [
                col for col in prepared.schema.numeric
                if col in x_test.columns
            ]
            if numeric:
                try:
                    rng = np.random.default_rng(seed + 10000 + fold)
                    stressed = x_test.copy(deep=True)
                    noise_scale = (
                        x_train[numeric].std(numeric_only=True)
                        .fillna(0.0).to_numpy(dtype=float)
                    )
                    original = stressed[numeric].to_numpy(dtype=float)
                    noisy = original + rng.normal(
                        0.0, 0.05 * noise_scale,
                        size=original.shape,
                    )
                    stressed.loc[:, numeric] = noisy
                    stressed_pred = np.asarray(
                        estimator.predict(stressed), dtype=float,
                    )
                    metrics["stress_noise_5pct_train_std_mae_shift"] = float(
                        np.mean(np.abs(stressed_pred - fold_predictions))
                    )
                    metrics["stress_noise_5pct_train_std_rmse"] = float(
                        regression_metrics(y_test, stressed_pred).rmse
                    )
                    metrics["stress_noise_evaluated"] = 1.0
                except (ValueError, TypeError, RuntimeError) as exc:
                    logger.warning(
                        "Model 1 holdout stress unavailable: algorithm=%s fold=%d error=%s",
                        algorithm, fold, exc,
                    )
                    metrics["stress_noise_evaluated"] = 0.0
            folds.append(
                {
                    "fold": fold,
                    **metrics,
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                    "validation_strategy": outer_strategy,
                    "resumed": False,
                }
            )
            if checkpoint:
                checkpoint.save_fold(
                    fold,
                    test_idx,
                    fold_predictions,
                    metrics,
                    params=params,
                )

            spinner.succeed(
                f"{label} · fold {fold}/{len(outer_splits)} | "
                f"RMSE={metrics['rmse']:.4f} | MAE={metrics['mae']:.4f} | "
                f"R²={metrics['r2']:.4f}"
            )
            logger.info(
                "algorithm=%s fold=%d strategy=%s n_train=%d n_test=%d "
                "params=%s metrics=%s",
                algorithm,
                fold,
                outer_strategy,
                len(train_idx),
                len(test_idx),
                params,
                metrics,
            )
        except BaseException:
            spinner.fail(
                f"{label} · fold {fold}/{len(outer_splits)} interrumpido"
            )
            raise
