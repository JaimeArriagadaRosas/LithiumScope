from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import pandas as pd

from lithiumscope.core.checkpoints import FoldCheckpointStore
from lithiumscope.core.scientific.checkpoint_identity import evaluation_identity
from lithiumscope.resources.detection.device import DeviceInfo
from lithiumscope.core.logging.session import get_logger
from lithiumscope.core.scientific_checks import (
    assert_group_isolation,
    assert_partition_coverage,
    assert_oof_complete,
)
from lithiumscope.core.scientific.regression_metrics import regression_metrics
from lithiumscope.model_1.validation.spatial import (
    materialize_regression_splits,
)
from lithiumscope.model_1.training.factory import build_pipeline, get_algorithm
from lithiumscope.model_1.training.optimization.optimizer import optimize_with_optuna
from lithiumscope.model_1.training.preprocessing.target_transform import (
    transform_label,
    validate_target_transform,
)
from lithiumscope.runtime.console_status import Spinner
from lithiumscope.model_1.training.execution.outer_folds import (
    OuterFoldEvidence, OuterFoldOptions, OuterFoldPartition, evaluate_outer_folds,
)

logger = get_logger("model_1.cv_runner")


@dataclass
class CVRunResult:
    algorithm: str
    label: str
    predictions: np.ndarray
    fold_table: pd.DataFrame
    overall_metrics: dict[str, float]
    elapsed_seconds: float
    final_params: dict
    target_transform: str = "identity"
    variant_id: str | None = None
    outer_splits: list | None = None
    robustness_table: pd.DataFrame | None = None
    preprocessing_records: list[dict] | None = None


def run_nested_cv(
    prepared,
    algorithm: str,
    device: DeviceInfo,
    config: dict,
    checkpoint_root: Path | None = None,
    *,
    target_transform: str = "identity",
    variant_id: str | None = None,
) -> CVRunResult:
    validation = config["validation"]
    optimization = config["optimization"]
    seed = int(validation["random_seed"])
    groups = (
        prepared.groups
        if bool(
            validation.get(
                "prefer_spatial_groups",
                False,
            )
        )
        else None
    )
    require_groups = bool(
        validation.get("prefer_spatial_groups", False)
        and not validation.get("allow_random_fallback", False)
    )
    if getattr(prepared, "outer_splits", None) is not None:
        outer_splits = getattr(prepared, "outer_splits")
        outer_strategy = getattr(prepared, "validation_strategy", None) or "unknown"
    else:
        outer_splits, outer_strategy = materialize_regression_splits(
            prepared.x,
            prepared.y,
            groups,
            int(validation["outer_folds"]),
            seed,
            require_groups=require_groups,
        )
    outer_splits = list(outer_splits)
    assert_partition_coverage(outer_splits, len(prepared.y))
    spec = get_algorithm(algorithm)
    target_transform = validate_target_transform(
        target_transform,
        prepared.y,
    )
    variant_id = (
        variant_id
        or f"{algorithm}__{target_transform}"
    )
    label = (
        spec.label
        if target_transform == "identity"
        else (
            f"{spec.label} · "
            f"{transform_label(target_transform)}"
        )
    )
    predictions = np.full(len(prepared.y), np.nan, dtype=float)
    folds: list[dict] = []
    robustness_records: list[dict] = []
    preprocessing_records: list[dict] = []
    started = time.perf_counter()
    trials = int(
        optimization.get("per_algorithm_trials", {}).get(
            algorithm,
            optimization.get("n_trials", 20),
        )
    )
    optimize = bool(optimization.get("enabled", True))
    pruning = optimization.get("pruning", {})
    from lithiumscope.model_1.training.preprocessing.treatment_policy import search_space
    treatment_profile=optimization.get("treatment_search_profile","legacy")
    if treatment_profile!="legacy" and (not optimize or target_transform!="identity" or groups is None):
        raise ValueError("Treatment search requires Optuna, grouped CV and identity variant")
    parameter_space=search_space(algorithm,spec.module.optuna_space,treatment_profile)
    if treatment_profile!="legacy":
        label += " · tratamientos TRAIN"
    checkpoint = (
        FoldCheckpointStore(checkpoint_root, variant_id, identity=evaluation_identity(checkpoint_root,model_group="model_1",
            x=prepared.x,y=prepared.y,groups=groups,splits=outer_splits,
            policy={"config":config,"variant":variant_id,"target_transform":target_transform,
                    "schema":prepared.schema,"device":device,"strategy":outer_strategy}))
        if checkpoint_root is not None
        else None
    )

    print(f"\n  ▶ {label}")
    logger.info(
        "Starting Model 1 algorithm=%s variant=%s target_transform=%s "
        "samples=%d checkpoint=%s",
        algorithm,
        variant_id,
        target_transform,
        len(prepared.y),
        checkpoint_root,
    )

    print(
        "    Validación externa: "
        f"{outer_strategy}"
    )

    evaluate_outer_folds(
        prepared, checkpoint,
        OuterFoldPartition(outer_splits, groups, outer_strategy, require_groups),
        OuterFoldOptions(
            algorithm=algorithm, device=device, label=label, optimize=optimize,
            pruning=pruning, seed=seed, spec=spec, target_transform=target_transform,
            trials=trials, validation=validation,
            physical_scenarios_enabled=bool(config.get("robustness", {}).get("physical_scenarios_enabled", True)),
            parameter_space=parameter_space,
        ),
        OuterFoldEvidence(predictions, folds, robustness_records, preprocessing_records),
    )

    assert_oof_complete(predictions, len(prepared.y))
    overall = regression_metrics(
        prepared.y,
        predictions,
    ).to_dict()
    fold_table = pd.DataFrame(folds)

    algorithm_cache = checkpoint.load_algorithm() if checkpoint else None
    if algorithm_cache is not None:
        final_params = dict(algorithm_cache.get("final_params", {}))
        previous_elapsed = float(algorithm_cache.get("elapsed_seconds", 0.0))
    else:
        final_params: dict = {}
        if optimize:
            spinner = Spinner(
                f"{label} · optimización final de hiperparámetros"
            ).start()
            try:
                def final_factory(candidate: dict):
                    return build_pipeline(
                        algorithm,
                        device,
                        seed,
                        prepared.schema,
                        candidate,
                        target_transform=target_transform,
                    )

                def final_progress(done: int, total: int) -> None:
                    spinner.update(
                        f"{label} · optimización final {done}/{total}"
                    )

                final_splits, _ = materialize_regression_splits(
                    prepared.x,
                    prepared.y,
                    groups,
                    int(validation["inner_folds"]),
                    seed + 5000,
                    require_groups=require_groups,
                )
                final_params = optimize_with_optuna(
                    final_factory,
                    parameter_space,
                    prepared.x,
                    prepared.y,
                    final_splits,
                    trials,
                    device=device,
                    groups=groups,
                    progress_callback=final_progress,
                    storage_path=(
                        checkpoint.root / "optuna_final.db"
                        if checkpoint
                        else None
                    ),
                    study_name=checkpoint.study_name(f"{algorithm}_final") if checkpoint else f"{algorithm}_final",
                    random_seed=seed + 5000,
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
                spinner.succeed(
                    f"{label} · hiperparámetros finales listos"
                )
            except BaseException:
                spinner.fail(
                    f"{label} · optimización final interrumpida"
                )
                raise
        previous_elapsed = 0.0

    if treatment_profile!="legacy" and "treatment_policy" not in final_params:
        raise ValueError("Missing selected treatment in final Optuna parameters")
    elapsed = previous_elapsed + (time.perf_counter() - started)
    if checkpoint and algorithm_cache is None:
        checkpoint.save_algorithm(
            {
                "algorithm": algorithm,
                "label": label,
                "variant_id": variant_id,
                "target_transform": target_transform,
                "final_params": final_params,
                "overall_metrics": overall,
                "elapsed_seconds": elapsed,
            }
        )

    return CVRunResult(
        algorithm=algorithm,
        label=label,
        predictions=predictions,
        fold_table=fold_table,
        overall_metrics=overall,
        elapsed_seconds=elapsed,
        final_params=final_params,
        target_transform=target_transform,
        variant_id=variant_id,
        outer_splits=outer_splits,
        robustness_table=pd.DataFrame(robustness_records),
        preprocessing_records=preprocessing_records,
    )
