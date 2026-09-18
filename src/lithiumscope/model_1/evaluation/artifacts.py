from __future__ import annotations

from pathlib import Path

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


def save_algorithm_artifacts(prepared, cv_result, run_context) -> dict[str, Path]:
    directory = run_context.figures / cv_result.algorithm
    directory.mkdir(parents=True, exist_ok=True)

    fold_path = run_context.tables / f"fold_metrics_{cv_result.algorithm}.csv"
    cv_result.fold_table.to_csv(fold_path, index=False)

    predictions = pd.DataFrame(
        {
            "row_index": prepared.y.index,
            "Li_icpms_real": prepared.y.to_numpy(),
            "Li_icpms_predicted": cv_result.predictions,
            "residual": prepared.y.to_numpy() - cv_result.predictions,
        }
    )
    prediction_path = run_context.tables / f"oof_predictions_{cv_result.algorithm}.csv"
    predictions.to_csv(prediction_path, index=False)

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
        "folds": fold_path,
        "predictions": prediction_path,
        "real_vs_predicted": real_vs_pred,
        "residuals": residuals,
    }
