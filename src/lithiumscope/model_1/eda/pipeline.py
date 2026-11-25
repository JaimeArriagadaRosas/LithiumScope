"""Model 1 EDA reuses the production preparation pipeline; never fits a model."""
from __future__ import annotations

from pathlib import Path
import pandas as pd

from lithiumscope.core.config import load_config
from lithiumscope.core.hashing import file_sha256
from lithiumscope.core.paths import RESULTS_DIR
from lithiumscope.core.scientific.eda_statistics import (
    category_frequencies, cohort_summary, feature_profiles, numeric_correlations,
)
from lithiumscope.model_1.pipeline import prepare_training_data
from lithiumscope.model_1.validation.spatial import materialize_regression_splits
from lithiumscope.core.scientific.fold_profiles import fold_profiles
from lithiumscope.core.scientific.spatial_audit import audit_spatial_folds
from lithiumscope.results.eda_artifacts import (
    complete_run, fail_run, find_compatible, new_run, signature,
)
from lithiumscope.results.eda_figures import save_figures
from lithiumscope.results.eda_protocol import fold_protocol, attach_training_eda


def run_eda(path: Path, *, output_root: Path | None = None, prepared=None, config=None) -> Path:
    prepared = prepare_training_data(path) if prepared is None else prepared
    cfg = load_config("model_1") if config is None else config
    if prepared.audit.dataset_sha256 != file_sha256(path):
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
        print(f"[OK] EDA descriptivo compatible reutilizado: {cached}")
        return cached
    directory = new_run(root)
    try:
        tables = directory / "tables"
        cols = [*prepared.schema.numeric, *prepared.schema.categorical]
        profiles = pd.concat([
            feature_profiles(prepared.raw_frame, cols, stage="raw"),
            feature_profiles(prepared.frame, cols, stage="prepared"),
        ], ignore_index=True)
        profiles.to_csv(tables / "feature_profiles.csv", index=False)
        prepared.audit.to_frame().to_csv(tables / "preparation_stages.csv", index=False)
        pd.DataFrame(prepared.audit.source_records).to_csv(tables / "stage_sources.csv", index=False)
        prepared.audit.exclusions_frame().to_csv(tables/"stage_exclusions.csv",index=False)
        pd.DataFrame(prepared.audit.column_records).to_csv(tables/"stage_column_missingness.csv",index=False)
        if prepared.coordinate_evidence is not None:
            prepared.coordinate_evidence.to_csv(tables/"coordinate_alias_audit.csv",index=False)
        reasons = []
        for stage, attr in (("05_target_validity", "target_exclusions"), ("06_quality_control", "qc_exclusions")):
            reasons.extend({"stage": stage, **record} for record in prepared.frame.attrs.get(attr, []))
        pd.DataFrame(reasons, columns=["stage", "row_index", "column", "original", "reason"]).to_csv(tables / "exclusion_reasons.csv", index=False)
        pd.DataFrame(prepared.audit.censoring, columns=["row_index", "source", "column", "original", "censor_type", "limit", "substituted"]).to_csv(tables / "censoring.csv", index=False)
        categories = pd.concat([category_frequencies(prepared.raw_frame, prepared.schema.categorical).assign(stage="raw"),
                                category_frequencies(prepared.frame, prepared.schema.categorical).assign(stage="prepared")], ignore_index=True)
        categories.to_csv(
            tables / "category_frequencies.csv", index=False)
        summary = cohort_summary(prepared.frame, prepared.target)
        summary.to_csv(tables / "cohort_summary.csv", index=False)
        correlations = numeric_correlations(prepared.frame, [*prepared.schema.numeric, prepared.target])
        correlations.to_csv(tables / "spearman_correlations.csv")
        save_figures(directory / "figures", prepared.frame, prepared.target,
                     profiles[profiles["stage"] == "prepared"], correlations)
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
        return complete_run(directory, signature_value=key, details={
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
    except Exception as exc:
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
