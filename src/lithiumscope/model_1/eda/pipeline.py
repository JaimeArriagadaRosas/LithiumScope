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
from lithiumscope.results.eda_artifacts import (
    complete_run, fail_run, find_compatible, new_run, signature,
)
from lithiumscope.results.eda_figures import save_figures


def run_eda(path: Path, *, output_root: Path | None = None) -> Path:
    prepared = prepare_training_data(path)
    cfg = load_config("model_1")
    root = output_root if output_root is not None else RESULTS_DIR / "model_1" / "eda"
    key = signature("model_1", path, cfg, prepared.frame)
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
        pd.DataFrame(prepared.audit.exclusions, columns=["stage", "row_index", "source", "sample_id"]).to_csv(
            tables / "stage_exclusions.csv", index=False)
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
        validation = cfg["validation"]
        fold_state = "not_available"
        if prepared.groups is not None and prepared.groups.nunique() >= int(validation["outer_folds"]):
            splits, strategy = materialize_regression_splits(prepared.x, prepared.y, prepared.groups,
                int(validation["outer_folds"]), int(validation["random_seed"]), require_groups=True)
            fold_summary, fold_features = fold_profiles(prepared.x, prepared.y, prepared.groups, splits)
            fold_summary.to_csv(tables / "outer_train_summary.csv", index=False)
            fold_features.to_csv(tables / "outer_train_features.csv", index=False)
            fold_state = strategy
        print(summary.to_string(index=False))
        return complete_run(directory, signature_value=key, details={
            "model_group": "model_1", "source_path": str(path), "source_sha256": file_sha256(path),
            "rows_raw": len(prepared.raw_frame), "rows_prepared": len(prepared.frame),
            "features": len(cols), "numeric_features": prepared.schema.numeric,
            "categorical_features": prepared.schema.categorical,
            "coordinate_report": prepared.coordinate_report,
            "target_validation": prepared.frame.attrs.get("target_validation", {}),
            "effective_config": cfg,
            "fold_profile_state": fold_state,
            "limitation": "Descriptive full cohort; not an independent development subset.",
        })
    except Exception as exc:
        fail_run(directory, exc)
        raise
