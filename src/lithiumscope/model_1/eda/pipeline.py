"""Model 1 EDA reuses the production preparation pipeline; never fits a model."""
from __future__ import annotations

from pathlib import Path

from lithiumscope.core.config import load_config
from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.logging.session import get_logger
from lithiumscope.core.paths import RESULTS_DIR
from lithiumscope.model_1.preparation.training_data import prepare_training_data
from lithiumscope.model_1.validation.spatial import materialize_regression_splits
from lithiumscope.core.scientific.fold_profiles import fold_profiles
from lithiumscope.core.scientific.spatial_audit import audit_spatial_folds
from lithiumscope.results.eda_artifacts import (
    complete_run, fail_run, find_compatible, new_run, signature,
)
from lithiumscope.model_1.eda.statistics import descriptive_tables
from lithiumscope.model_1.eda.artifacts import write_descriptive_tables
from lithiumscope.model_1.eda.visualization import render_profiles
from lithiumscope.results.eda_protocol import fold_protocol, attach_training_eda


logger = get_logger("model_1.eda")


def run_eda(path: Path, *, output_root: Path | None = None, prepared=None, config=None) -> Path:
    logger.info("M1_EDA_STARTED source=%s", path)
    prepared = prepare_training_data(path) if prepared is None else prepared
    cfg = load_config("model_1") if config is None else config
    if prepared.audit.dataset_sha256 != file_sha256(path):
        logger.error("M1_EDA_SOURCE_CHANGED source=%s", path)
        raise ValueError("M1 prepared source changed before EDA")
    validation = cfg["validation"]
    splits, strategy, protocol = None, "not_available", None
    if prepared.groups is not None and prepared.groups.nunique() >= int(validation["outer_folds"]):
        if prepared.outer_splits is None:
            splits, strategy = materialize_regression_splits(prepared.x, prepared.y, prepared.groups,
                int(validation["outer_folds"]), int(validation["random_seed"]), require_groups=True)
        elif prepared.validation_strategy == "group_kfold_spatial":
            splits, strategy = prepared.outer_splits, prepared.validation_strategy
        if splits is not None:
            protocol = fold_protocol(prepared.x, prepared.y, prepared.groups, splits, strategy=strategy)
    root = output_root if output_root is not None else RESULTS_DIR / "model_1" / "eda"
    key = signature("model_1", path, cfg, prepared.frame, protocol=protocol)
    cached = find_compatible(root, key)
    if cached:
        logger.info("M1_EDA_REUSED run=%s", cached.name)
        print(f"[OK] EDA descriptivo compatible reutilizado: {cached}")
        return cached
    directory = new_run(root)
    try:
        tables = directory / "tables"
        cols = [*prepared.schema.numeric, *prepared.schema.categorical]
        calculated=descriptive_tables(prepared)
        write_descriptive_tables(calculated,prepared,tables)
        render_profiles(directory/"figures",prepared,calculated)
        summary=calculated["cohort_summary"]
        fold_state = "not_available"
        if splits is not None:
            fold_summary, fold_features = fold_profiles(prepared.x, prepared.y, prepared.groups, splits)
            fold_summary.to_csv(tables / "outer_train_summary.csv", index=False)
            fold_features.to_csv(tables / "outer_train_features.csv", index=False)
            spatial_summary, spatial_rows = audit_spatial_folds(prepared.frame, prepared.groups, splits,
                minimum_distance_m=float(validation.get("minimum_distance_m", 0)))
            spatial_summary.to_csv(tables / "outer_spatial_summary.csv", index=False)
            spatial_rows.to_csv(tables / "outer_spatial_nearest.csv", index=False)
            fold_state = strategy
        print(summary.to_string(index=False))
        result = complete_run(directory, signature_value=key, details={
            "model_group": "model_1", "source_path": str(path), "source_sha256": file_sha256(path),
            "tabular_source": prepared.raw_frame.attrs.get("tabular_source", {}),
            "rows_raw": len(prepared.raw_frame), "rows_prepared": len(prepared.frame),
            "features": len(cols), "numeric_features": prepared.schema.numeric,
            "categorical_features": prepared.schema.categorical,
            "coordinate_report": prepared.coordinate_report,
            "target_validation": prepared.frame.attrs.get("target_validation", {}),
            "effective_config": cfg,
            "fold_profile_state": fold_state, "development_protocol": protocol,
            "limitation": "Descriptive full cohort; not an independent development subset.",
        })
        logger.info("M1_EDA_COMPLETED run=%s rows=%d features=%d fold_state=%s", directory.name, len(prepared.frame), len(cols), fold_state)
        return result
    except Exception as exc:
        logger.exception("M1_EDA_FAILED run=%s", directory.name)
        fail_run(directory, exc)
        raise


def record_training_eda(path, prepared, config, context):
    if prepared.validation_strategy != "group_kfold_spatial":
        context.tracker.event("eda_development_unavailable", reason="nonspatial validation")
        return
    protocol = fold_protocol(prepared.x, prepared.y, prepared.groups, prepared.outer_splits,
                             strategy=prepared.validation_strategy)
    directory = run_eda(path, prepared=prepared, config=config)
    attach_training_eda(directory, context, protocol)
    context.tracker.event("eda_evidence_verified", eda_run_id=directory.name)
