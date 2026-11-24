from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd

from lithiumscope.model_1.evaluation.plots import (
    save_correlation_heatmap,
    save_missingness,
    save_pipeline_rows,
    save_real_vs_predicted,
    save_residuals,
    save_target_distribution,
)


def save_dataset_artifacts(prepared, run_context) -> dict[str, Path | None]:
    audit = prepared.audit.to_frame()
    audit_path = run_context.tables / "pipeline_steps.csv"
    audit.to_csv(audit_path, index=False)
    pd.DataFrame(prepared.audit.source_records).to_csv(run_context.tables/"stage_sources.csv",index=False)
    pd.DataFrame(prepared.audit.column_records).to_csv(run_context.tables/"stage_column_missingness.csv",index=False)
    prepared.audit.exclusions_frame().to_csv(run_context.tables/"stage_exclusions.csv",index=False)
    if prepared.coordinate_evidence is not None:
        prepared.coordinate_evidence.to_csv(run_context.tables/"coordinate_alias_audit.csv",index=False)

    describe_path = run_context.tables / "dataset_describe.csv"
    prepared.frame.describe(include="all").transpose().reset_index(names="variable").to_csv(describe_path, index=False)

    target_path = save_target_distribution(prepared.y, run_context.figures / "target_distribution.png")
    missing_path = save_missingness(prepared.raw_frame, run_context.figures / "missing_values.png")
    pipeline_path = save_pipeline_rows(audit, run_context.figures / "pipeline_sample_counts.png")
    heatmap_path = save_correlation_heatmap(
        prepared.frame,
        prepared.target,
        run_context.figures / "correlation_heatmap.png",
    )

    correlations = prepared.frame.select_dtypes(include=[np.number]).corr(numeric_only=True)
    correlation_path = run_context.tables / "numeric_correlations.csv"
    correlations.to_csv(correlation_path)

    return {
        "audit": audit_path,
        "describe": describe_path,
        "target_distribution": target_path,
        "missing_values": missing_path,
        "pipeline_rows": pipeline_path,
        "correlation_heatmap": heatmap_path,
        "correlations": correlation_path,
    }


def save_algorithm_tables(prepared, cv_result, run_context) -> dict[str, Path]:
    artifact_key = (
        cv_result.variant_id
        or cv_result.algorithm
    )
    fold_path = run_context.tables / f"fold_metrics_{artifact_key}.csv"
    cv_result.fold_table.to_csv(fold_path, index=False)

    from lithiumscope.model_1.evaluation.oof_evidence import regression_oof_table
    splits = cv_result.outer_splits if cv_result.outer_splits is not None else prepared.outer_splits
    predictions = regression_oof_table(prepared, cv_result.predictions, splits)
    prediction_path = run_context.tables / f"oof_predictions_{artifact_key}.csv"
    predictions.to_csv(prediction_path, index=False)

    artifacts={"folds": fold_path, "predictions": prediction_path}
    from lithiumscope.model_1.evaluation.high_lithium import save_high_lithium_diagnostics
    confusion_path=run_context.tables/f"high_lithium_confusion_{artifact_key}.csv"
    strata_path=run_context.tables/f"high_lithium_residuals_{artifact_key}.csv"
    save_high_lithium_diagnostics(predictions,confusion_path,strata_path)
    artifacts.update(high_lithium_confusion=confusion_path,high_lithium_residuals=strata_path)
    if cv_result.robustness_table is not None and not cv_result.robustness_table.empty:
        stress_path=run_context.tables / f"robustness_{artifact_key}.csv"
        cv_result.robustness_table.to_csv(stress_path,index=False)
        artifacts["robustness"]=stress_path
    if cv_result.preprocessing_records:
        preprocess_path=run_context.tables / f"preprocessing_{artifact_key}.json"
        preprocess_path.write_text(json.dumps(cv_result.preprocessing_records,indent=2,allow_nan=False),encoding='utf-8')
        artifacts['preprocessing']=preprocess_path
    return artifacts


def save_algorithm_figures(prepared, cv_result, run_context) -> dict[str, Path]:
    directory = run_context.figures / (cv_result.variant_id or cv_result.algorithm)
    directory.mkdir(parents=True, exist_ok=True)
    real_vs_pred = save_real_vs_predicted(
        prepared.y,
        cv_result.predictions,
        directory / "real_vs_predicted.png",
        title=f"{cv_result.label} — Real vs. predicho",
    )
    residuals = save_residuals(
        prepared.y,
        cv_result.predictions,
        directory / "residuals.png",
        title=f"{cv_result.label} — Residuos",
    )
    return {
        "real_vs_predicted": real_vs_pred,
        "residuals": residuals,
    }

def save_algorithm_artifacts(prepared, cv_result, run_context) -> dict[str, Path]:
    return {**save_algorithm_tables(prepared, cv_result, run_context),
            **save_algorithm_figures(prepared, cv_result, run_context)}
