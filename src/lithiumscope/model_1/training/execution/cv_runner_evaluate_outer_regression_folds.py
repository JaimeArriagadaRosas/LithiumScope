"""Extracted responsibility; original public entrypoint remains stable."""
from __future__ import annotations


def _evaluate_outer_regression_folds(algorithm, checkpoint, device, folds, groups, label, optimize, outer_splits, outer_strategy, predictions, prepared, pruning, require_groups, seed, spec, target_transform, trials, validation, robustness_records, preprocessing_records, physical_scenarios_enabled):
    from lithiumscope.model_1.training.execution.cv_runner import Spinner, assert_group_isolation, build_pipeline, logger, materialize_regression_splits, np, optimize_with_optuna, regression_metrics
    from lithiumscope.model_1.training.preprocessing.spatial_tabnet import fit_regression_estimator, early_stopping_evidence
    from lithiumscope.core.logging.context import scientific_log_context
    for fold, (train_idx, test_idx) in enumerate(
        outer_splits,
        start=1,
    ):
        with scientific_log_context(algorithm=algorithm,fold=fold,stage="outer_cv"):
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
                robustness_records.extend(cached.get('robustness',[]))
                if cached.get('preprocessing') is not None:
                    preprocessing_records.append({'fold':fold,**cached['preprocessing']})
                fold_predictions = np.asarray(
                    cached["predictions"],
                    dtype=float,
                )
                predictions[test_idx] = fold_predictions
                metrics = {**cached["metrics"], **regression_metrics(prepared.y.iloc[test_idx], fold_predictions).to_dict()}
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
                        device=device,
                        groups=groups_train,
                        progress_callback=progress,
                        storage_path=(
                            checkpoint.root / f"optuna_fold_{fold:02d}.db"
                            if checkpoint
                            else None
                        ),
                        study_name=checkpoint.study_name(f"{algorithm}_fold_{fold:02d}") if checkpoint else f"{algorithm}_fold_{fold:02d}",
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
                fit_regression_estimator(estimator,x_train,y_train,groups_train)
                epoch_evidence=early_stopping_evidence(estimator)
                from lithiumscope.model_1.training.preprocessing.evidence import preprocessing_evidence
                prep_evidence=preprocessing_evidence(estimator,x_train,x_test)
                if prep_evidence is not None:
                    preprocessing_records.append({'fold':fold,**prep_evidence})
                fold_predictions = estimator.predict(x_test)
                predictions[test_idx] = fold_predictions
                metrics = regression_metrics(
                    y_test,
                    fold_predictions,
                ).to_dict()
                if prep_evidence is not None:
                    metrics['transformed_feature_count']=prep_evidence['feature_count']
                    metrics['train_all_missing_feature_count']=len(prep_evidence['train_all_missing_columns'])
                train_predictions = estimator.predict(x_train)
                metrics.update({f"train_{k}":v for k,v in regression_metrics(y_train,train_predictions).to_dict().items()})
                if epoch_evidence is not None:
                    metrics.update({"tabnet_best_epoch":epoch_evidence["best_epoch"],"tabnet_refit_n":epoch_evidence["refit_n"],
                                    "tabnet_validation_n":epoch_evidence["n_validation"],"tabnet_validation_groups":len(epoch_evidence["validation_groups"])})
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
                scenario_records=[]
                if physical_scenarios_enabled:
                    from lithiumscope.model_1.evaluation.robustness import evaluate_stress
                    scenario_records=evaluate_stress(estimator,x_test,y_test,fold_predictions,prepared.schema,fold=fold,seed=seed)
                    robustness_records.extend(scenario_records)
                    metrics['stress_physical_scenarios_ok']=sum(row['status']=='ok' for row in scenario_records)
                    metrics['stress_physical_scenarios_failed']=sum(row['status']=='failed' for row in scenario_records)
                    metrics['stress_physical_scenarios_unavailable']=sum(row['status']=='unavailable' for row in scenario_records)
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
                        early_stopping=epoch_evidence,
                        preprocessing=prep_evidence,
                        robustness=scenario_records,
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
